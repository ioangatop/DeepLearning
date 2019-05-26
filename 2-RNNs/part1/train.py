################################################################################
# MIT License
# 
# Copyright (c) 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

################################ Accuracy ######################################
def accuracy(predictions, targets):
    """ Computes the prediction accuracy, i.e. the average of correct predictions
        of the network.
    """
    # Get the the probability and the predicted class for each image
    top_p, top_class = predictions.topk(1, dim=1)

    # Check if the predicted classes match the labels
    equals = top_class == targets.view(*top_class.shape)

    # Calculate the percentage of correct predictions
    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

    return accuracy
################################################################################

################################## Train #######################################
def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    # if GPU was chosen, check if CUDA is available
    if str(config.device) != "cpu":
        if not torch.cuda.is_available():
            print('\n* GPU was selected but CUDA is not available.\nTraining on CPU ...')
            device = torch.device("cpu")
        else:
            print('\nCUDA is available!  Training on GPU ...')
            device = torch.device(config.device)
    else:
        print('\nTraining on GPU ...')
        device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim,
                           config.num_hidden, config.num_classes, config.batch_size, device)
    else:
        model = LSTM(config.input_length, config.input_dim,
                     config.num_hidden, config.num_classes, config.batch_size, device)

    # Print Configuration
    print("Model Type: {!s:5} Input Length: {!s:5} Learning Rate: {}\n"
          .format(config.model_type, config.input_length, config.learning_rate))

    # Initialize model
    model = torch.nn.DataParallel(model).to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    train_loss, train_accuracy, train_steps = [], [], []

    # Enable train mode
    model.train()

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # move tensors to GPU, if enabled
        batch_targets = batch_targets.long().to(device)
        batch_inputs = batch_inputs.to(device)

        # Forward pass
        predictions = model(batch_inputs)

        # Calculate loss
        loss = criterion(predictions, batch_targets)

        # Back-propagate
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        # ANSWER: `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # ref: https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Update weights
        optimizer.step()

        # Clear weights gradients
        optimizer.zero_grad()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            # Store accuracy and loss
            train_steps.append(step)
            train_loss.append(loss.item())
            train_accuracy.append(accuracy(predictions, batch_targets))

            if step % 100 == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                          datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                          config.train_steps, config.batch_size, examples_per_second,
                          train_accuracy[-1], train_loss[-1]))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655

            # Save Train and Test accuracies and losses
            file_name = str(config.model_type) + '_' + str(config.input_length) + '.npz'
            np.savez(file_name,
                     train_steps=train_steps,
                     train_accuracy=train_accuracy,
                     model_type=config.model_type,
                     input_length=config.input_length)

            break

    print('Done training.')
################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=20, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
