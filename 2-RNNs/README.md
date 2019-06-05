# 3 - Generative Models

[![License](http://img.shields.io/:license-mit-blue.svg)](LICENSE)

<p align="center">
  <img src="readme_imgs/generative_model.gif" width="150" />
</p>
<!-- <p align="center">
    Generate text with LSTM.
</p> -->

## Description
We studied and implemented three of the most famous and powerful generative models, namely __Variational Auto Encoders__ (VAEs), __Generative Adversarial Networks__ (GANs) and __Generative Normalizing Flows__ (NFs). We analyzed both from theoretical and practical spectrum, we presented their mathematical framework and results for practical implementations.

- [Assignment](3-Generative_models/assignment_3.pdf), [report](3-Generative_models/gatopoulos_report_assignment3.pdf) and [code](3-Generative_models/code)


## VAEs

<p align="center">
  <img src="readme_imgs/vae_pipeline.png" width="400" />
</p>
<p align="center">
    VAE manifold.
</p>

A VAE is a latent variable model that leverages the flexibility of Neural Networks (NN) in order to learn/specify a latent variable model.

<p align="center">
  <img src="readme_imgs/vae_manifold.gif" width="200" />
  <img src="readme_imgs/vae_elbo.png" width="350" />
</p>
<p align="center">
    VAE manifold.
</p>

## GANs

<p align="center">
  <img src="readme_imgs/gan_pipeline.png" width="400" />
</p>
<p align="center">
    VAE manifold.
</p>

Generative Adversarial Networks (GAN) are a type of deep generative models. Similar to VAEs, GANs can generate images that mimick images from the dataset by sampling an encoding from a noise distribution. In constract to VAEs, in vanilla GANs there is no inference mechanism to determine an encoding or latent vector that corresponds to a given data point (or image).

<p align="center">
  <img src="readme_imgs/gan_training.gif" width="500" />
</p>
<p align="center">
    GAN training progress.
</p>


## NFs
Similar to VAEs, Flow-based generative models have a distribution over the latent variables. Unlike VAEs, there is an exact correspondence (not a distribution) between a single datapoint and a latent representation.