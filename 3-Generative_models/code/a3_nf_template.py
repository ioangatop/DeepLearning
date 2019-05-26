import os
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from datasets.mnist import mnist


def log_prior(x, k=256):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    # log probability
    logp = -0.5 * (x ** 2 + np.log(2 * np.pi))
    # Sum them batch-wize
    logp = logp.view(x.size(0), -1).sum(-1)
    return logp

def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """
    sample = torch.randn(size).float().to(device)
    return sample

def get_bpd(log_p, dimentions=28*28):
    """
    bpd = (nll_val / num_pixels) / numpy.log(2).

    log_p: log probability
    dimentions: dimentions (resolution) of image
    """
    return ((-log_p / dimentions) / math.log(2)).mean().item()


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.nn = torch.nn.Sequential(
            nn.Linear(c_in, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, 2*c_in)
            )

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, z, log_det_J, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        # First-half masked z
        z_m1 = self.mask * z
        # Second-half masked z
        z_m2 = (1 - self.mask) * z
        # Get the scale and transpose from the neural net
        s, t = self.nn(z_m1).chunk(2, dim=1)
        s = torch.tanh(s)

        if not reverse:
            z = z_m1 + (1 - self.mask) * (z * torch.exp(s) + t)
            log_det_J += ((1 - self.mask) * s).sum(dim=1)
        else:
            z = z_m1 + (1 - self.mask) * ((z - t) * torch.exp(-s))
            log_det_J -= ((1 - self.mask) * s).sum(dim=1)

        return z, log_det_J

class Flow(nn.Module):
    def __init__(self, shape, n_flows):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()
        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape, n_flows):
        super().__init__()
        self.flow = Flow(shape, n_flows)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, z):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example
        log_pz = log_prior(z)
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        # Sample (random or fixed) z from latent space
        z = sample_prior((n_samples,) + self.flow.z_shape)
        ldj = torch.zeros(z.size(0), device=z.device)

        # Reverse opperation
        z, logdet = self.flow.forward(z, ldj, reverse=True)
        # Normalize the output
        z, logdet = self.logit_normalize(z, logdet, reverse=True)
        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """
    avg_bpd = 0.0
    if model.training:
        for step, (imgs) in enumerate(data):
            # Move data to device
            imgs = imgs[0].to(device)
            # Calculate log_px
            log_px = model.forward(imgs)
            # Calculate loss: NLLLoss
            loss = -log_px.mean()
            # Calculate loss: bpd
            bpd = get_bpd(log_px)
            # Clear accumalate gradients
            optimizer.zero_grad()
            # Perform backprop
            loss.backward()
            # Update weights
            optimizer.step()
            # Update average bpd variable
            avg_bpd += bpd
            print(bpd)
    else:
        for step, (imgs) in enumerate(data):
            # Move data to device
            imgs = imgs[0].to(device)
            # Calculate log_px
            log_px = model(imgs)
            # Calculate loss: NLLLoss
            loss = -log_px.mean()
            # Calculate loss: bpd
            bpd = get_bpd(log_px)
            # Update average bpd
            avg_bpd += bpd

    # Get the average
    avg_bpd /= step

    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename, dpi=None):
    # Init subfigure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    # Set axis
    ax.plot(train_curve, label='Train BPD')
    ax.plot(val_curve, label='Validation BPD')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('BPD')
    ax.set_title("Real NVP Test and Validation BPD")

    # No boundaries
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set len of axis
    max_ = max(max(train_curve), max(val_curve))
    min_ = min(min(train_curve), min(val_curve))
    max_ = round(max_) + 1
    min_ = max(round(min_) - 0.5, 0)

    # Plot black lines
    y_axis = np.arange(min_, max_, 0.5)
    for y in y_axis:
        ax.plot(range(0, ARGS.epochs), y * np.ones(ARGS.epochs), "--", lw=0.5, color="black", alpha=0.3)
    ax.legend(loc='upper right')

    fig.tight_layout()

    # Save figure
    fig.savefig(ARGS.gen_image_path+filename+'.png')
    plt.close(fig)


def main():
    # Create output image directory
    os.makedirs(ARGS.gen_image_path, exist_ok=True)

    # Get data --ignore test split
    data = mnist()[:2]

    # Init model
    model = nn.DataParallel(Model(shape=[28*28], n_flows=ARGS.n_flows).to(device))
    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr, betas=(ARGS.b1, ARGS.b2), eps=1e-07)
    # Clip grads
    nn.utils.clip_grad_norm(model.parameters(), ARGS.max_grad_norm)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        # Run one epoch and get the bpds
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        # Save bpds to lists
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)

        print('Epoch [{:4d}/{:4d}] | Train BPD: {:6.2f} | Validation BPD: {:6.2f}'.format(
            epoch+1, ARGS.epochs, train_bpd, val_bpd))

        # --------------------------------------------------------------------
        #  Plot samples from model during training.
        # --------------------------------------------------------------------
        # Samle image
        X = model.module.sample(ARGS.sample_size)
        # Save sampled image
        save_image(X.data.view(-1, 1, 28, 28),
                   ARGS.gen_image_path+'nf_{}.png'.format(epoch+1),
                   nrow=int(math.sqrt(ARGS.sample_size)), normalize=True)

        # save_bpd_plot(train_curve, val_curve, 'nfs_bpd_pdf_{}'.format(epoch+1))

    save_bpd_plot(train_curve, val_curve, 'nfs_bpd_pdf_{}'.format(ARGS.epochs+2), dpi=300)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    ## Training parameters
    PARSER.add_argument('--epochs', default=40+1, type=int,
                        help='max number of epochs')
    PARSER.add_argument('--n_flows', default=4, type=int,
                        help='Number of flows')

    # Learning rate
    PARSER.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    PARSER.add_argument("--b1", type=float, default=0.9,
                        help="momentum; beta1 in Adam optimizer.")
    PARSER.add_argument("--b2", type=float, default=0.999,
                        help="decay; beta2 in Adam optimizer.")
    # scheduler
    PARSER.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    PARSER.add_argument('--learning_rate_step', type=int, default=1,
                        help='Learning rate step')

    PARSER.add_argument('--max_grad_norm', type=float, default=100.,
                        help='Max gradient norm for clipping')

    # Experiments parameters
    PARSER.add_argument('--gen_image_path', type=str, default="./images_nfs/",
                        help='Output path for generated images.')
    PARSER.add_argument('--sample_size', type=int, default=25,
                        help='The number of the generated images.')
    PARSER.add_argument('--sample_fixed_dist', type=bool, default=True,
                        help='The output images are generated by a fixed latent distribution.')

    ARGS = PARSER.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main()
