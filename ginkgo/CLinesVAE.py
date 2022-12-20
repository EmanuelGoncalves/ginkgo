import ginkgo
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot as uplot
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt

# check if cuda is available and change devide accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Drug Response dataset
class CLinesDataset(data.Dataset):
    def __init__(self, gexp_csv_file, prot_csv_file):
        # Read csv files
        self.df_gexp = pd.read_csv(gexp_csv_file, index_col=0).T
        self.df_prot = pd.read_csv(prot_csv_file, index_col=0).T

        # Union samples
        self.samples = list(set(self.df_gexp.index).union(set(self.df_prot.index)))
        self.df_gexp = self.df_gexp.reindex(index=self.samples)
        self.df_prot = self.df_prot.reindex(index=self.samples)

        # Parse the data
        self.x_gexp, self.scaler_gexp = self.process_df(self.df_gexp)
        self.x_prot, self.scaler_prot = self.process_df(self.df_prot)

        # Datasets list
        self.views = [self.x_gexp, self.x_prot]

    def process_df(self, df):
        # Normalize the data using StandardScaler
        scaler = StandardScaler()
        x = scaler.fit_transform(df)
        x = np.nan_to_num(x, copy=False)

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float)

        return x, scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.x_gexp[idx], self.x_prot[idx]


# Define the VAE model
class ClinesVAE(nn.Module):
    def __init__(self, views, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.view_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(v.shape[1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim * 2),
                )
                for v in views
            ]
        )

        # Decoder layers
        self.view_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, v.shape[1]),
                )
                for v in views
            ]
        )

    def poe(self, mus, logvars):
        # formula (prior var = 1): var_joint = inv(inv(var_prior) + sum(inv(var_modalities)))
        logvar_joint = torch.sum(
            torch.stack([1.0 / torch.exp(log_var) for log_var in logvars]), dim=0
        )
        logvar_joint = torch.log(1.0 / logvar_joint)

        # formula (prior mu = 0): mu_joint = (mu_prior*inv(var_prior) + sum(mu_modalities*inv(var_modalities))) * var_joint
        mu_joint = torch.sum(
            torch.stack([mu / torch.exp(log_var) for mu, log_var in zip(mus, logvars)]),
            dim=0,
        )
        mu_joint = mu_joint * torch.exp(logvar_joint)

        return mu_joint, logvar_joint

    def reparameterize(self, mu, sd):
        std = torch.exp(0.5 * sd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, views):
        mus, log_vars = [], []

        for v, e in zip(views, self.view_encoders):
            h = e(v)
            mus.append(h[:, : self.latent_dim])
            log_vars.append(h[:, self.latent_dim :])

        return mus, log_vars

    def decode(self, z):
        return [d(z) for d in self.view_decoders]

    def forward(self, views):
        # Encode the input and then decode it
        mus, logvars = self.encode(views)

        # Product of experts
        mu_joint, logvar_joint = self.poe(mus, logvars)

        # Reparameterize
        z = self.reparameterize(mu_joint, logvar_joint)

        # Decode
        x_hat = self.decode(z)

        return x_hat, mu_joint, logvar_joint


def mse_kl(views_hat, views, mu_joint, logvar_joint, alpha=0.1):
    # Number of samples in the batch
    n = views[0].size(0)

    # Compute the MSE loss
    mse_loss = sum(
        [
            torch.nn.functional.mse_loss(x_hat, x, reduction="sum")
            for x, x_hat in zip(views, views_hat)
        ]
    )

    # Compute the KL loss
    kl_loss = -0.5 * torch.sum(1 + logvar_joint - mu_joint.pow(2) - logvar_joint.exp())

    # Compute the total loss
    loss = (mse_loss + alpha * kl_loss) / n

    return loss, mse_loss / n, (alpha * kl_loss) / n


def clines_labels(clabel):
    if clabel == "Haematopoietic and Lymphoid":
        return "Haem"
    elif clabel == "Lung":
        return "Lung"
    elif clabel == "Large Intestine":
        return "LargeIntestine"
    elif clabel == "Skin":
        return "Skin"
    elif clabel == "Breast":
        return "Breast"
    elif clabel == "Bone":
        return "Bone"
    else:
        return "Other"


if __name__ == "__main__":
    # Load the first dataset
    clines_db = CLinesDataset(
        gexp_csv_file="data/clines/transcriptomics.csv",
        prot_csv_file="data/clines/proteomics.csv",
    )

    # Create a data loader for the dataset
    dataloader = data.DataLoader(clines_db, batch_size=128, shuffle=True)

    # Create the VAE model
    model = ClinesVAE(clines_db.views, 1024, 2)

    # Train
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    for epoch in range(epochs):
        for views in dataloader:
            x_hat, mu_joint, logvar_joint = model.forward(views)

            vae_loss, reconstruction_loss, kl_loss = mse_kl(
                x_hat, views, mu_joint, logvar_joint, 0.0001
            )
            loss = vae_loss + reconstruction_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch: {epoch}, Loss: {loss.item()}, VAE Loss: {vae_loss.item()}, Reconstruction Loss: {reconstruction_loss.item()}, KL Loss: {kl_loss.item()}"
            )

    # Plot latent space
    ss = pd.read_csv("data/clines/cmp_model_list_20221102.csv", index_col=0)

    z = pd.DataFrame(
        model.poe(*model.encode(clines_db.views))[0].detach().numpy(),
        index=clines_db.samples,
    )

    # z_umap = pd.DataFrame(
    #     umap.UMAP(n_neighbors=15, min_dist=0.1, metric="correlation").fit_transform(z),
    #     index=z.index,
    # )

    df = pd.concat(
        [
            z,
            ss["tissue"].reindex(z.index).apply(lambda v: clines_labels(v)),
        ],
        axis=1,
    )

    palette = dict(
        Other="gray",
        Haem="red",
        Lung="blue",
        LargeIntestine="green",
        Skin="orange",
        Breast="brown",
        Bone="purple",
    )

    plt.subplots(1, 1, figsize=(4, 4))
    for l in palette.keys():
        plt.scatter(
            df.query(f"tissue == '{l}'")[0],
            df.query(f"tissue == '{l}'")[1],
            label=l,
            c=palette[l],
            zorder=2,
            alpha=0.7,
            linewidths=0,
            s=5,
        )
    plt.legend(loc="upper right")
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("VAE Latent space transcriptomics (UMAP)")

    plt.savefig(
        "plots/clines/clines_vae_umap.pdf",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
