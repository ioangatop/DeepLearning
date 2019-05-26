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

import torch
import torch.nn as nn

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, drop_prob=0.5, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        # Save model confgs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.drop_prob = drop_prob
        self.device = device

        # Define the LSTM
        self.lstm = nn.LSTM(input_size=vocabulary_size, 
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers, 
                            dropout=drop_prob,
                            batch_first=True)
        
        # Define the Fully Connected output layer
        self.fc = nn.Linear(lstm_num_hidden, vocabulary_size)

    def one_hot_encode(self, x, vocab):
        # Initialize the the encoded tensor
        one_hot = torch.zeros((torch.mul(*x.shape), vocab), dtype=torch.float32).to(self.device)
        # Fill the appropriate elements with ones
        one_hot[torch.arange(one_hot.shape[0]), x.flatten()] = 1.
        # Finally reshape it to get back to the original tensor
        one_hot = one_hot.reshape((*x.shape, vocab))
        return one_hot

    def init_hidden(self, batch_size):
        '''
        Initializes hidden state.
        Create two new tensors with sizes n_layers x batch_size x n_hidden,
        initialized to zero, for hidden state and cell state of LSTM.

        # Within the batch loop, we detach the hidden state from its history;
        # this time setting it equal to a new tuple variable because an LSTM has
        # a hidden state that is a tuple of the hidden and cell states.

        Comments:
            'next': returns the first parameter from the class.
            'new' : constructs a new tensor of the same data type (as the first parameter).
        '''
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_num_layers, batch_size, self.lstm_num_hidden).zero_().to(self.device),
                    weight.new(self.lstm_num_layers, batch_size, self.lstm_num_hidden).zero_().to(self.device))
        return hidden

    def forward(self, x):
        """ Forward pass of model
            ----------------------
            x.shape = batch x sequ x vocabulary_size --> 128x50x83
            lstm_output.shape = batch x sequ x hidden
        """
        if self.training:
            # Initialized lstm hidden layers to zero, for hidden state and cell state of LSTM.
            # Otherwise we'd backprop through the entire training history
            self.lstm_hidden = self.init_hidden(x.shape[0])
        # x to one-hot vector
        x = self.one_hot_encode(x, self.vocabulary_size)
        # Recurrent pass
        lstm_output, self.lstm_hidden = self.lstm(x, self.lstm_hidden)
        # Stack LSTM output
        lstm_output = lstm_output.contiguous().view(-1, self.lstm_num_hidden)
        # Forward pass from the fc layer
        predictions = self.fc(lstm_output)

        return predictions
