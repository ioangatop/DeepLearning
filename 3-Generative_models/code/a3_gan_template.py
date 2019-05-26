import argparse
import os

import math
import numpy as np

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim, dropout_prob):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        hidden = [128, 256, 512, 1024, 28*28]

        net = []
        net.append(nn.Linear(latent_dim, hidden[0]))
        for i in range(len(hidden)-1):
            if i != 0:  # Do not apply Batch Norm in the first layer
                net.append(nn.BatchNorm1d(hidden[i]))
            net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            if i != 0:  # Do not apply Dropout in the first layer
                net.append(nn.Dropout(dropout_prob))
            net.append(nn.Linear(hidden[i], hidden[i+1]))
        net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(self, z):
        """ Forward pass: Generate images from z
        """
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, dropout_prob):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(nn.Linear(28*28, 512),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Dropout(dropout_prob),
                                 nn.Linear(512, 256),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Linear(256, 1),
                                 nn.Sigmoid()
                                )

    def forward(self, img):
        """ Forward pass: Returns discriminator score for img
        """
        # Flatten image
        img = img.view(-1, 28*28)
        return self.net(img)


def get_labels(batch_size, label_smoothing, flipped_labbels):
    """ Creates labels for the loss with the real and the face sample.
        Also, flip labbels if it is True
    """
    if label_smoothing:
        # Label Smoothing -- Salimans et. al. 2016
        if flipped_labbels:
            real_labels = torch.ones(batch_size, 1).to(device) * torch.FloatTensor(1).uniform_(0.0, 0.3).to(device)
            fake_labels = torch.ones(batch_size, 1).to(device) * torch.FloatTensor(1).uniform_(0.7, 1.2).to(device)
        else:
            real_labels = torch.ones(batch_size, 1).to(device) * torch.FloatTensor(1).uniform_(0.7, 1.2).to(device)
            fake_labels = torch.ones(batch_size, 1).to(device) * torch.FloatTensor(1).uniform_(0.0, 0.3).to(device)
    else:
        if flipped_labbels:
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
        else:
            real_labels = torch.zeros(batch_size, 1).to(device)
            fake_labels = torch.ones(batch_size, 1).to(device)

    return real_labels, fake_labels

def interpolate(generator):
    # Interpolation steps
    n = 7
    # Sample two digits
    z_1 = np.random.normal(-.7, .7, ARGS.latent_dim)
    z_2 = np.random.normal(.7, .7, ARGS.latent_dim)

    # Initialize the interpolation space
    interpolation_space = np.linspace(z_1, z_2, n+2)

    digits_list = []
    for digit in interpolation_space:
        z =  torch.from_numpy(digit).float().to(device) * torch.ones((ARGS.latent_dim)).to(device)
        digits_list.append(z)

    # Stack tensors
    z = torch.stack(digits_list, dim=0).to(device)
    # Generate images
    fake_imgs = generator(z)
    # Save generate images
    save_image(fake_imgs.data.view(-1, 1, 28, 28),
               'interpolate_digits.png', nrow=n+2, normalize=True)
    print('\nProduced interpolation between two digits. Saved as: interpolate_digits.png\n')

def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    # Move models to device
    discriminator = discriminator.to(device)
    generator = generator.to(device)

    # Get dimention of the latent space
    latent_dim = ARGS.latent_dim

    # Initialise criterion loss
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(ARGS.n_epochs):
        avg_d_loss, avg_g_loss = 0.0, 0.0
        for i, (imgs, _) in enumerate(dataloader):
            # Get batch size
            batch_size = imgs.shape[0]

            # Get labels for the loss with the real and the fake sample.
            real_labels, fake_labels = get_labels(batch_size,
                                                  ARGS.label_smoothing,
                                                  ARGS.flipped_labbels)

            # Mode data to device
            imgs = imgs.to(device)

            # Sample from laten dist and generate fake images
            z = torch.randn((batch_size, ARGS.latent_dim)).to(device)
            fake_imgs = generator(z)

            # ================================
            #      TRAIN THE DISCRIMINATOR
            # ================================
            # Clear accumalate gradients
            optimizer_D.zero_grad()

            # Forward pass of discriminator
            d_real = discriminator(imgs)
            # Calculate real loss
            d_real_loss = criterion(d_real, real_labels)

            ### Forward pass of discriminator
            # detach fake_imgs, otherwise we could not backprop
            d_fake = discriminator(fake_imgs)
            # d_fake = discriminator(fake_imgs.detach())

            # Calculate fake loss
            d_fake_loss = criterion(d_fake, fake_labels)

            # Calculate total loss
            d_loss = d_real_loss + d_fake_loss
            avg_d_loss += d_loss

            # Perform backprop
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            # ================================
            #       TRAIN THE GENERATOR
            # ================================
            # Clear accumulate gradients
            optimizer_G.zero_grad()

            # Compute the discriminator losses on fake images
            d_fake = discriminator(fake_imgs)
            g_loss = criterion(d_fake, real_labels)
            avg_g_loss += g_loss

            # Perform backprop
            g_loss.backward()
            optimizer_G.step()

            # ================================
            #        GENERATE SAMPLES
            # ================================
            batches_done = epoch * len(dataloader) + i
            if batches_done % ARGS.save_interval == 0:
                # Print loss stats
                print('Step/Epoch [{:5d}/{:5d}] | D_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                    i+1, epoch+1, d_loss.item(), g_loss.item()))

                # Generator into evaluation mode
                if ARGS.eval:
                    generator.eval()

                if ARGS.sample_fixed_dist:
                    # Generate fake images from the fixed distribution
                    fake_imgs = generator(FIXED_Z)
                else:
                    # Sample some random latent variables
                    z = torch.randn((ARGS.sample_size, latent_dim)).to(device)
                    # Generate fake images from the fixed distribution
                    fake_imgs = generator(z)

                # Save generated images
                save_image(fake_imgs.data.view(-1, 1, 28, 28),
                           ARGS.gen_image_path+'gan_{}.png'.format(batches_done),
                           nrow=int(math.sqrt(ARGS.sample_size)), normalize=True)

                # Back to train mode
                generator.train()

def main():
    # Create output image directory
    os.makedirs(ARGS.gen_image_path, exist_ok=True)

    # Load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize([0.5], [0.5])])),
        batch_size=ARGS.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = nn.DataParallel(Generator(ARGS.latent_dim, ARGS.dropout_G))
    discriminator = nn.DataParallel(Discriminator(ARGS.dropout_D))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=ARGS.lr, betas=(ARGS.b1, ARGS.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=ARGS.lr, betas=(ARGS.b1, ARGS.b2))

    if ARGS.generator_load is None:
        print_(ARGS)

        # Start training
        train(dataloader, discriminator, generator, optimizer_G, optimizer_D)
        print('\nDone training.')

        # Save Generator
        torch.save(generator.state_dict(), "mnist_generator.pt")
    else:
        # Load Generator
        generator.load_state_dict(torch.load(ARGS.generator_load, map_location=str(device)))
        print('\nGenerator loaded.\n')
        # Put it on evaluation mode
        generator.eval()

        # Linear interpolation between two digits
        interpolate(generator)

def print_(ARGS):
    print('Training GAN on MNIST\n')
    print('Training epochs: {}'.format(ARGS.n_epochs))
    print('Dimensionality of latent space: {}'.format(ARGS.latent_dim))
    print('Dropout probability on the Discriminator: {}'.format(ARGS.dropout_D))
    print('Dropout probability on the Generator: {}'.format(ARGS.dropout_G))
    print('Label Smoothing: {}'.format(str(ARGS.label_smoothing)))
    print('Flipped Labbels: {}'.format(str(ARGS.flipped_labbels)))
    print('Training on: {}\n'.format(str(device)))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    # Training parameters
    PARSER.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    PARSER.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    PARSER.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    PARSER.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    PARSER.add_argument("--b1", type=float, default=0.5,
                        help="momentum; beta1 in Adam optimizer.")
    PARSER.add_argument("--b2", type=float, default=0.999,
                        help="decay; beta2 in Adam optimizer.")
    PARSER.add_argument('--dropout_D', type=float, default=0.2,
                        help='Dropout probability on the Discriminator.')
    PARSER.add_argument('--dropout_G', type=float, default=0.2,
                        help='Dropout probability on the Generator.')
    PARSER.add_argument('--label_smoothing', type=bool, default=True,
                        help='Label Smoothing.')
    PARSER.add_argument('--flipped_labbels', type=bool, default=True,
                        help='Flipped Labbels.')

    # Experiments parameters
    PARSER.add_argument('--save_interval', type=int, default=100,
                        help='save every SAVE_INTERVAL iterations')
    PARSER.add_argument('--sample_size', type=int, default=25,
                        help='The number of the generated images.')
    PARSER.add_argument('--sample_fixed_dist', type=bool, default=False,
                        help='The output images are generated by a fixed latent distribution.')
    PARSER.add_argument('--gen_image_path', type=str, default="./images_gan/",
                        help='Output path for generated images.')
    PARSER.add_argument('--eval', type=bool, default=True,
                        help='Evaluation mode On/Off when sampling.')

    # Load generator
    PARSER.add_argument('--generator_load', type=str, default=None,
                        help='Directory path of generator.')

    ARGS = PARSER.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if ARGS.sample_fixed_dist:
        # Get some fixed data for sampling. These are images that are held
        # constant throughout training, and allow us to inspect the model's performance
        FIXED_Z = torch.randn((ARGS.sample_size, ARGS.latent_dim)).to(device)

    main()
