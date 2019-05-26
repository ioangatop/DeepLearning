# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse
import random

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

#################################################################################

################################ Accuracy #######################################
def accuracy(predictions, targets):
    """ Computes the prediction accuracy of the network.
    """
    # Get the the probability and the predicted class for each image
    top_p, top_class = predictions.topk(1, dim=1)

    # Check if the predicted classes match the labels
    equals = top_class == targets.view(*top_class.shape)

    # Calculate the percentage of correct predictions
    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

    return accuracy
################################################################################

################################ Save model ####################################
def save_model(epoch, step, model):
    """ Saves LSTM model
    """
    model_name = 'lstm' + '_epoch_' + str(epoch) + '_step_' + str(step)
    path = './part2/models/' + model_name

    checkpoint = {'batch_size': model.batch_size,
                  'seq_length': model.seq_length,
                  'vocabulary_size': model.vocabulary_size,
                  'lstm_num_hidden': model.lstm_num_hidden,
                  'lstm_num_layers': model.lstm_num_layers,
                  'drop_prob': model.drop_prob,
                  'device': model.device,
                  'state_dict': model.state_dict()}

    with open(path, 'wb') as f:
        torch.save(checkpoint, f)
################################################################################

############################### Load a model ###################################
def load_model(model_name):
    """ Loads LSTM model
    """
    with open(model_name, 'rb') as f:
        checkpoint = torch.load(f)  

    loaded_model = TextGenerationModel(checkpoint['batch_size'],
                                       checkpoint['seq_length'],
                                       checkpoint['vocabulary_size'],
                                       checkpoint['lstm_num_hidden'],
                                       checkpoint['lstm_num_layers'],
                                       checkpoint['drop_prob'],
                                       checkpoint['device']).to(checkpoint['device'])

    loaded_model.load_state_dict(checkpoint['state_dict'])
    return loaded_model

################################################################################

################################ Save txt ######################################
def save_txt(gen_char, config, path, epoch, step, method):
    """ Saves summaries in .txt form in path
    """
    txt_name = 'text' + '_epoch_' + str(epoch) + '_step_' + str(step) + '-' + method +'.txt'
    txt_path = path + txt_name
    text_file = open(txt_path, "w")
    text_file.write("\n" + str('-')*25 + "\n" +
                    "Model Config" + "\n" +
                    str('-')*25 + "\n" +
                    "Epoch: " + str(epoch) + "\n" +
                    "Step: " + str(step) + "\n" +
                    "batch_size: " + str(config.batch_size) + "\n" +
                    "seq_length: " + str(config.seq_length) + "\n" +
                    "lstm_num_hidden: " + str(config.lstm_num_hidden) + "\n" +
                    "lstm_num_layers: " + str(config.lstm_num_layers) + "\n" +
                    "dropout: " + str(1 - config.dropout_keep_prob) + "\n" +
                    "method: " + str(method) + "\n" +
                    str('-')*25 + "\n"
                    )
    text_file.write(''.join(gen_char))
    text_file.close()
################################################################################

############################# tempering_sample #################################
def tempering_sample(model, dataset, beta, config, device, epoch, step, print_=True, char=None):
    method = str(beta)

    if char is None:
        # Randomly sample the first character
        sample_char = random.choices([*dataset._char_to_ix.keys()], k=1)
        char = dataset._char_to_ix[sample_char[0]]

    # Detach hidden state from history
    model.lstm_hidden = model.init_hidden(torch.Tensor([[char]]).long().to(device).shape[0])

    gen_char = []
    for _ in range(config.seq_length):
        # Mova char to long Tensor
        char = torch.Tensor([[char]]).long().to(device)
        # Forward pass
        outputs = model(char)
        outputs = outputs/beta
        # Select the next character with some element of randomness
        pdf = torch.distributions.Categorical(logits=outputs)
        char = pdf.sample() 
        char = char.cpu().numpy().squeeze().item()
        gen_char.append(dataset.convert_to_string([char]))

    # Print generated text
    if print_:
        print_lins = str('-')*(129)
        print('Method: betta ', method)
        print(print_lins +'\n', ''.join(gen_char), '\n'+ print_lins +'\n')

    # Save generated text as .txt
    save_txt(gen_char, config, config.summary_path, epoch, step, method)
################################################################################

############################### greedy_sample ##################################
def greedy_sample(model, dataset, config, device, epoch, step, print_=True, char=None):
    method = "greedy"

    if char is None:
        # Randomly sample the first character
        sample_char = random.choices([*dataset._char_to_ix.keys()], k=1)
        char = dataset._char_to_ix[sample_char[0]]

    # Detach hidden state from history
    model.lstm_hidden = model.init_hidden(torch.Tensor([[char]]).long().to(device).shape[0])

    gen_char = []
    for _ in range(config.seq_length):
        # Mova char to long Tensor
        char = torch.Tensor([[char]]).long().to(device)
        # Forward pass
        outputs = model(char)
        # Get pdf of next-character scores
        pdf = F.softmax(outputs, dim=1).data
        # Get the most probable char
        prob, char = pdf.topk(1, dim=1)
        char = char.cpu().numpy().squeeze().item()
        gen_char.append(dataset.convert_to_string([char]))

    # Print generated text
    if print_:
        print_lins = str('-')*(129)
        print('Method: ', method)
        print(print_lins +'\n', ''.join(gen_char), '\n'+ print_lins +'\n')

    # Save generated text as .txt
    save_txt(gen_char, config, config.summary_path, epoch, step, method)
################################################################################

############################### gen_from_word ##################################
def gen_from_word(word, model, dataset, config, device, epoch, step, sampling_meth, T, print_=True):

    """ Takes a word and generates text out of it.
        We want to prime the network and can build up a hidden state

    ### Some notes:
    # The output of our RNN is from a fully-connected layer
    # and it outputs a distribution of next-character scores.

    # To actually get the next character, we apply a softmax
    # function, which gives us a probability distribution that
    # we can then sample to predict the next character.

    # Our predictions come from a categorical probability
    # distribution over all the possible characters.
    """
    assert sampling_meth in ('top_k', 'beta')

    method = word + "_" + str(sampling_meth)
    top_k = 5
    beta = 0.5

    # Input a word and convert it into a number
    chars = [dataset._char_to_ix[char] for char in word]
    gen_char = [char for char in word]

    # Detach hidden state from history
    model.lstm_hidden = model.init_hidden(torch.Tensor([[chars[0]]]).long().to(device).shape[0])

    # Forward pass for every char:
    # -- we want to prime the network and can build up a hidden state
    for char in chars:
        char = torch.Tensor([[char]]).long().to(device)
        outputs = model(char)

    if sampling_meth == 'top_k':
        ##### Sample next char with top_k method ####
        # Get pdf of next-character scores
        pdf = F.softmax(outputs, dim=1).data
        # Get the top_k characters
        top_probs, top_chars = pdf.topk(top_k)
        # Select the next character with some element of randomness
        top_ch = top_chars.cpu().numpy().squeeze()
        p = top_probs.cpu().numpy().squeeze()
        # Normalize probabilities
        p /= p.sum()
        # Sample next char
        char = np.random.choice(top_ch, p=p)
    else:
        outputs = outputs/beta
        # Select the next character with some element of randomness
        pdf = torch.distributions.Categorical(logits=outputs)
        char = pdf.sample() 
        char = char.cpu().numpy().squeeze().item()

    # set last character as the first input to the generating model
    gen_char.append(dataset.convert_to_string([char]))

    for _ in range(T):
        # Mova char to long Tensor
        char = torch.Tensor([[char]]).long().to(device)
        # Forward pass
        outputs = model(char)

        if sampling_meth == 'top_k':
            ### topk method ###
            # Get pdf of next-character scores
            pdf = F.softmax(outputs, dim=1).data
            # Get the top_k characters
            top_probs, top_chars = pdf.topk(top_k)
            # Select the next character with some element of randomness
            top_ch = top_chars.cpu().numpy().squeeze()
            p = top_probs.cpu().numpy().squeeze()
            # Normalize probabilities
            p /= p.sum()
            # Sample next char
            char = np.random.choice(top_ch, p=p)
        else:
            ### betta method ###
            outputs = outputs/beta
            # Select the next character with some element of randomness
            pdf = torch.distributions.Categorical(logits=outputs)
            char = pdf.sample() 
            char = char.cpu().numpy().squeeze().item()

        gen_char.append(dataset.convert_to_string([char]))

    # Print generated text
    if print_:
        print_lins = str('-')*(129)
        print('Generate from word: ', word, ' using ', sampling_meth ,' sampling.')
        print(print_lins +'\n', ''.join(gen_char), '\n'+ print_lins +'\n')

    # Save generated text as .txt
    save_txt(gen_char, config, config.summary_path, epoch, step, method)

################################################################################

################################## Train #######################################

def train(config):
    # Create output generated images directory (if it does not already exists)
    os.makedirs('./generated_text/', exist_ok=True)
    os.makedirs('./models/', exist_ok=True)

    os.makedirs('./part2/generated_text/', exist_ok=True)
    os.makedirs('./part2/models/', exist_ok=True)

    # Initialize the device which to run the model on
    # if GPU was chosen, check if CUDA is available
    if str(config.device) != "cpu":
        if not torch.cuda.is_available():
            print('\n* GPU was selected but CUDA is not available.\nTraining on CPU ...\n')
            device = torch.device("cpu")
        else:
            print('\n* CUDA is available!  Training on GPU ...\n')
            device = torch.device(config.device)
    else:
        print('\n* Training on GPU ...\n')
        device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    model = TextGenerationModel(config.batch_size,
                                config.seq_length,
                                dataset.vocab_size,
                                lstm_num_hidden=config.lstm_num_hidden,
                                lstm_num_layers=config.lstm_num_layers,
                                drop_prob=1.0 - config.dropout_keep_prob,
                                device=device).to(device)

    # Setup the loss, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step,
                                                gamma=config.learning_rate_decay)

    train_accuracy, train_loss = [], []
    eval_steps, eval_loss, eval_accuracy, = [], [], []

    for epoch in range(config.epochs):
        # Print current epoch
        print('\n',str('-')*(56), 'epoch: {}/{}'.format(epoch+1, config.epochs), str('-')*(56))

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Enable train mode
            model.train()

            # Only for time measurement of step through network
            t1 = time.time()

            ################################################################
            # batch_inputs.shape = batch_size x seq_lenght dimentions
            batch_inputs = torch.stack(batch_inputs, dim=1).to(device)
            batch_targets = torch.stack(batch_targets, dim=1).to(device)

            # Update batch size 
            # -- in case that the last batch size is less than the confg. one
            config.batch_size = batch_inputs.shape[0]

            # Clear accumulated gradients
            optimizer.zero_grad() 

            # Forward pass
            predictions = model(batch_inputs)

            # Calculate loss
            loss = criterion(predictions, batch_targets.view(config.batch_size*config.seq_length))

            # Store train accuracy and loss
            train_loss.append(loss.item())
            train_accuracy.append(accuracy(predictions, batch_targets))

            # Back-propagate
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

            # Update weights and scheduler
            optimizer.step()
            scheduler.step(loss.item())
            ################################################################

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                print("[{}] Train Step {:04f}/{:04f}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                          datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                          config.train_steps, config.batch_size, examples_per_second,
                          train_accuracy[-1], train_loss[-1]))

            if step % config.sample_every == 0:
                # Generate sentences by sampling from the model
                print("\n* Sampling...\n")

                # Model into evaluation mode
                model.eval()

                # If summaries are prinded between the training
                print_ = False

                # Tempering Sampling
                betas = [0.5, 1, 2]
                for beta in betas:
                    tempering_sample(model, dataset, beta, config, device, epoch, step, print_)

                # Greedy Sampling
                greedy_sample(model, dataset, config, device, epoch, step, print_)

                # Bonus part: Generate sentence given a sentence
                sentence = 'They run into the train.'
                T = 2000
                sampling_methodes = ['top_k', 'beta']
                for sampling_meth in sampling_methodes:
                    gen_from_word(sentence, model, dataset, config, device, epoch, step, sampling_meth, T, print_)

                sentence = 'Anna'
                T = 2000
                sampling_methodes = ['top_k', 'beta']
                for sampling_meth in sampling_methodes:
                    gen_from_word(sentence, model, dataset, config, device, epoch, step, sampling_meth, T, print_)

                # Save the trained model -- Checkpoint
                # save_model(epoch, step, model)

                # Save loss and accuracy
                eval_steps.append(step)
                eval_loss.append(train_loss[-1])
                eval_accuracy.append(train_accuracy[-1])
                np.savez('lstm.npz', eval_steps=eval_steps, eval_loss=eval_loss, eval_accuracy=eval_accuracy)

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')

    # Save the trained model -- Checkpoint
    save_model(epoch, step, model)
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='./part2/books/anna.txt', required=False,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=512,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")

    # Training params
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0,
                        help='--')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./part2/generated_text/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=500,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=1000,
                        help='How often to sample from the model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Load a trained LSTM model')

    config = parser.parse_args()

    # Train the model
    train(config)
