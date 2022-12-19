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
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, index_col=0).T

        # Normalize the data using StandardScaler
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df.values)
        self.X = np.nan_to_num(self.X, copy=False)

        self.X = torch.tensor(self.X, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# Define the VAE model
class ClinesVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_sd = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h1 = self.encoder(x)

        mu = self.encoder_mu(h1)
        sd = self.encoder_sd(h1)

        z = self.reparameterize(mu, sd)

        kl = -0.5 * torch.sum(1 + sd - mu.pow(2) - sd.exp())

        return z, mu, sd, kl

    def reparameterize(self, mu, sd):
        std = torch.exp(0.5 * sd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode the input and then decode it
        z, mu, sd, kl = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, sd, kl


def mse_kl(x_hat, x, mu, logvar, alpha=0.1):
    mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (mse_loss + alpha * kl_loss) / x.size(0)
    return loss, mse_loss / x.size(0), (alpha * kl_loss) / x.size(0)


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
    gexp = CLinesDataset("data/clines/transcriptomics.csv")

    # Create a data loader for the dataset
    dataloader = data.DataLoader(gexp, batch_size=128, shuffle=True)

    # Create the VAE model
    model = ClinesVAE(gexp.X.shape[1], 2048, 32)

    # Train
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    for epoch in range(epochs):
        for x in dataloader:
            x_hat, mu, sd, kl = model.forward(x)

            vae_loss, reconstruction_loss, kl_loss = mse_kl(x_hat, x, mu, sd, 0.0001)
            loss = vae_loss + reconstruction_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Plot latent space
    ss = pd.read_csv("data/clines/cmp_model_list_20221102.csv", index_col=0)

    z = pd.DataFrame(model.encode(gexp.X)[0].detach().numpy(), index=gexp.df.index)

    z_umap = pd.DataFrame(
        umap.UMAP(n_neighbors=15, min_dist=0.1, metric="correlation").fit_transform(z),
        index=z.index,
    )

    df = pd.concat(
        [
            z_umap,
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
