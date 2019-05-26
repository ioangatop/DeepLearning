import os
import argparse

import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2*z_dim)
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean, log_var = self.nn(input).chunk(2, dim=1)

        return mean, log_var


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.fc_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = self.fc_decoder(input)
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        # Flatten image: batch_size x 784
        input = input.view(-1, 784)

        ### Pass though the encoder
        z_mu, z_log_var = self.encoder(input)

        ### Reparametrization Trick
        epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5*z_log_var)*epsilon

        ### Pass though the decoder
        input_approx = self.decoder(z)

        ### Average Negative ELBO
        # KL divergence
        kl = - 0.5*torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
        # Binary Cross Entropy loss
        bce = F.binary_cross_entropy(input_approx, input, reduction='sum')
        # Get the average wrt the number of batches
        average_negative_elbo = (kl + bce)/input.shape[0]

        return average_negative_elbo

    def sample(self, n_samples, fixed=True):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        ## Samle from latent space
        # Use every time fixed points or random
        if fixed:
            z = FIXED_Z
        else:
            z = torch.randn((n_samples, self.z_dim)).to(device)

        # Pass it though the decoder to generate an image
        im_means = self.decoder(z)
        # Apply bernoulli to get the sampled images
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = 0.0
    if model.training:
        for step, (imgs) in enumerate(data):
            # Move data to device
            imgs = imgs.to(device)
            # Clear accumalate gradients
            optimizer.zero_grad()
            # Calculate loss
            elbo = model(imgs)
            # Perform backprop
            elbo.backward()
            # Update weights
            optimizer.step()
            # Update average ELBO variable
            average_epoch_elbo += elbo.item()
    else:
        for step, (imgs) in enumerate(data):
            # Move data to device
            imgs = imgs.to(device)
            # Calculate loss
            elbo = model(imgs)
            # Update average ELBO
            average_epoch_elbo += elbo.item()

    # Get the average
    average_epoch_elbo /= step

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    # Put model on training mode
    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    # Switch on evaluation mode
    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename, dpi=None):
    # Init subfigure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    # Set axis
    ax.plot(train_curve, label='Train ELBO')
    ax.plot(val_curve, label='Validation ELBO')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('ELBO')
    ax.set_title("VAE's Test and Validation ELBO")

    # No boundaries
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set len of axis
    max_ = max(max(train_curve), max(val_curve))
    max_ = max_ - max_%10 + 10
    min_ = min(min(train_curve), min(val_curve))
    min_ = min_ - min_%10 - 10

    # Plot black lines
    y_axis = np.arange(min_, max_, 10)
    for y in y_axis:
        ax.plot(range(0, ARGS.epochs), y * np.ones(ARGS.epochs), "--", lw=0.5, color="black", alpha=0.3)
    ax.legend(loc='upper right')

    fig.tight_layout()

    # Save figure
    fig.savefig(filename+'.png')
    plt.close(fig)

def plot_manifold(model, epoch, n_manifold=19):
    """
    Saves picture of the manifold learned space on epoch

    model: VAE model
    epoch: current epoch
    n_manifold: dimention of the manifold grid. Results into nxn digits
    """

    # Set the grid space
    z1 = torch.from_numpy(norm.ppf(np.linspace(0.01, 0.99, n_manifold))).float().to(device)
    z2 = torch.from_numpy(norm.ppf(np.linspace(0.01, 0.99, n_manifold))).float().to(device)
    # Get the grid
    manifold_grid = torch.stack(torch.meshgrid(z1, z2))
    manifold_grid = manifold_grid.permute(2, 1, 0)
    # Stack tensos in form (batch_size x z_dim)
    manifold_grid = manifold_grid.contiguous().view(-1, ARGS.zdim)

    # Pass it though the decoder to generate images
    manifold_imgs = model.module.decoder(manifold_grid)

    # Save the manifold
    save_image(manifold_imgs.data.view(-1, 1, 28, 28),
               ARGS.gen_image_path+'vae_manifold_{}.png'.format(epoch+1), nrow=n_manifold)

def plot_(model, data, epoch):
    """ Display a 2D plot of the digit classes in the latent space
    """
    model.eval()
    traindata, valdata = data
    z_list = []
    for _, (imgs) in enumerate(valdata):
        # Move data to device
        imgs=imgs.to(device)
        # Flatten image: batch_size x 784
        imgs = imgs.view(-1, 784)
        # Pass though the encoder
        z_mu, z_log_var = model.encoder(imgs)
        z_list.append(z_mu.detach().numpy())
    z = np.vstack(z_list)

    ### Init subfigure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    # No boundaries
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.scatter(z[:, 0], z[:, 1], c=np.ones_like(z[:, 0]),
               alpha=.4, s=3**2, cmap='viridis')
    # fig.colorbar()
    # Save figure
    fig.savefig('PLOT_'+str(epoch)+'.png')
    plt.close(fig)

def main():
    # Create output image directory
    os.makedirs(ARGS.gen_image_path, exist_ok=True)

    # Get data --ignore test split
    data = bmnist()[:2]

    # Init model
    model = nn.DataParallel(VAE(z_dim=ARGS.zdim).to(device))
    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        # Run one epoch and get the avg ELBO loss
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        # Save the looses to lists
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)

        print('Epoch [{:4d}/{:4d}] | Train ELBO: {:6.2f} | Validation ELBO: {:6.2f}'.format(
            epoch+1, ARGS.epochs, train_elbo, val_elbo))

        # --------------------------------------------------------------------
        #  Plot samples from model during training.
        # --------------------------------------------------------------------
        # Sample and generate images
        sampled_ims, im_means = model.module.sample(ARGS.n_samples, ARGS.fixed_samples)

        # Save generated images in binary
        save_image(sampled_ims.data.view(-1, 1, 28, 28), ARGS.gen_image_path+'vae_'+str(ARGS.zdim)+'d_{}.png'
                   .format(epoch+1), nrow=round(math.sqrt(ARGS.n_samples)))

        # Save generated images mean
        save_image(im_means.data.view(-1, 1, 28, 28), ARGS.gen_image_path+'mean_vae_'+str(ARGS.zdim)+'d_{}.png'
                   .format(epoch+1), nrow=round(math.sqrt(ARGS.n_samples)))

    # --------------------------------------------------------------------
    #  Plot the learned data manifold after training
    #  if the dimention of the latent space is 2.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        plot_manifold(model, epoch)

    # Save ELBO loss plot
    save_elbo_plot(train_curve, val_curve, ARGS.gen_image_path+'vae_elbo_pdf_{}'.format(ARGS.epochs), dpi=300)

    # Save train and validation
    np.savez('vae_losses.npz', epochs=ARGS.epochs, train_loss=train_curve, val_loss=val_curve)

    print('Done training.')

def print_(ARGS):
    print('Training VAE on binary-MNIST\n')
    print('Training epochs: {}'.format(ARGS.epochs))
    print('Dimensionality of latent space: {}'.format(ARGS.zdim))
    print('Training on: {}\n'.format(str(device)))

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    # Training
    PARSER.add_argument('--epochs', default=40+1, type=int,
                        help='max number of epochs')
    PARSER.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    # Experiments
    PARSER.add_argument('--n_samples', default=25, type=int,
                        help='Number of generated samples.')
    PARSER.add_argument('--fixed_samples', type=bool, default=False,
                        help='Sample from fixed points of latent space.')
    PARSER.add_argument('--gen_image_path', type=str, default="./images_vae/",
                        help='Output path for generated images.')

    ARGS = PARSER.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get fixed sample from latent space (if enabled)
    if ARGS.fixed_samples:
        FIXED_Z = torch.randn((ARGS.n_samples, ARGS.zdim)).to(device)

    print_(ARGS)

    main()
