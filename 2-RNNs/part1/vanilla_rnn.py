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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        """
        Input of RNN: batch_size x seq_length x encode
        seq_length: window that we are going to feed the NN at every pass
        encode: 10, if one-hot or 1 if encoded
        """
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_hidden = num_hidden
        self.device = torch.device(device)

        # Initialize hidden layer
        self.h_init = nn.Parameter(torch.zeros(self.batch_size, self.num_hidden), requires_grad=False)

        # Parameters initialization w/ Xavier uniform
        stdv_h = 1.0 / math.sqrt(num_hidden)
        stdv_c = 1.0 / math.sqrt(num_classes)
        self.Wx = torch.nn.Parameter(torch.randn(input_dim, num_hidden).uniform_(-stdv_h, stdv_h))
        self.Wh = torch.nn.Parameter(torch.randn(num_hidden, num_hidden).uniform_(-stdv_c, stdv_c))
        self.Wo = torch.nn.Parameter(torch.randn(num_hidden, num_classes).uniform_(-stdv_c, stdv_c))

        # Biases initialization w/ zeros
        self.bh = torch.nn.Parameter(torch.zeros(num_hidden))
        self.bo = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        """ Input x: batch_size x length_of_sequence
        """
        # Initialiaze hidden state
        hidden = self.h_init

        # Recurent pass
        for step in range(self.seq_length):
            hidden = nn.Tanh()(x[:, step].unsqueeze(-1) @ self.Wx + hidden @ self.Wh + self.bh)

        # Calculate predictions
        p = hidden @ self.Wo + self.bo
        return p
