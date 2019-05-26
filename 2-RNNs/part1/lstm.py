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

import torch
import torch.nn as nn
import math

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim

        # Initialize cell memory layer and hidden layer
        self.h_init = nn.Parameter(torch.zeros(num_hidden, batch_size), requires_grad=False)
        self.c_init = nn.Parameter(torch.zeros(num_hidden, batch_size), requires_grad=False)

        stdv_h = 1.0 / math.sqrt(num_hidden)
        stdv_c = 1.0 / math.sqrt(num_classes)

        # LSTM parameters initialization w/ Xavier uniform
        self.W_gx = torch.nn.Parameter(torch.randn(num_hidden,  input_dim).uniform_(-stdv_h, stdv_h))
        self.W_gh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden).uniform_(-stdv_h, stdv_h))
        self.W_ix = torch.nn.Parameter(torch.randn(num_hidden,  input_dim).uniform_(-stdv_h, stdv_h))
        self.W_ih = torch.nn.Parameter(torch.randn(num_hidden, num_hidden).uniform_(-stdv_h, stdv_h))
        self.W_fx = torch.nn.Parameter(torch.randn(num_hidden,  input_dim).uniform_(-stdv_h, stdv_h))
        self.W_fh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden).uniform_(-stdv_h, stdv_h))
        self.W_ox = torch.nn.Parameter(torch.randn(num_hidden,  input_dim).uniform_(-stdv_h, stdv_h))
        self.W_oh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden).uniform_(-stdv_h, stdv_h))

        # LSTM biases initialization w/ zeros
        self.b_g = torch.nn.Parameter(torch.zeros(num_hidden))
        self.b_i = torch.nn.Parameter(torch.zeros(num_hidden))
        self.b_f = torch.nn.Parameter(torch.zeros(num_hidden))
        self.b_o = torch.nn.Parameter(torch.zeros(num_hidden))

        # From hidden to output parameters and biases initialization w/ Xavier uniform and w/ zeros respectively
        self.W_ph = torch.nn.Parameter(torch.randn(num_classes, num_hidden).uniform_(-stdv_c, stdv_c))
        self.b_p = torch.nn.Parameter(torch.zeros(num_classes, 1))

    def forward(self, x):
        """
        x is (batch, input_size) --> (input_size x batch)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        # x.shape: (batch x input_size) --> (input_size x batch)
        x = x.permute(1, 0) 

        # Initialize hidden and cell states
        h = self.h_init
        c = self.c_init

        # Recurent pass
        for step in range(self.seq_length):
            # Compute gates
            g = nn.Tanh()(self.W_gx @ x[step, :].unsqueeze(0) + self.W_gh @ h + self.b_g)
            i = nn.Sigmoid()(self.W_ix @ x[step, :].unsqueeze(0) + self.W_ih @ h + self.b_i)
            f = nn.Sigmoid()(self.W_fx @ x[step, :].unsqueeze(0) + self.W_fh @ h + self.b_f)
            o = nn.Sigmoid()(self.W_ox @ x[step, :].unsqueeze(0) + self.W_oh @ h + self.b_o)

            # Update hidden and cell layers
            c = g*i + c*f
            h = nn.Tanh()(c)*o

        # Calculate predictions
        p = self.W_ph @ h + self.b_p

        return p.t()
