"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()

    channels = [n_channels, 64, 128, 256, 256, 512, 512, 512, 512]
    net = []
    k = 0

    ###################### CNN ######################
    for _ in range(2):
      net.append(nn.Conv2d(channels[k], channels[k+1], kernel_size=3, stride=1, padding=1))
      net.append(nn.BatchNorm2d(channels[k+1]))
      net.append(nn.ReLU())
      net.append(nn.MaxPool2d(3, stride=2, padding=1))
      k+=1

    for _ in range(3):
      for _ in range(2):
        net.append(nn.Conv2d(channels[k], channels[k+1], kernel_size=3, stride=1, padding=1))
        net.append(nn.BatchNorm2d(channels[k+1]))
        net.append(nn.ReLU())
        k+=1
      net.append(nn.MaxPool2d(3, stride=2, padding=1))

    net.append(nn.AvgPool2d(1, stride=1, padding=0))

    self.cnn_model = nn.Sequential(*net)


    ############ Fully Connected Layer ##############
    net = []
    net.append(nn.Linear(512, n_classes))
    net.append(nn.Softmax(dim=1))

    self.fc_model = nn.Sequential(*net)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Output of CNN
    cnn_out = self.cnn_model(x) 
    # flatten image input
    cnn_out = cnn_out.view(cnn_out.shape[0], -1)
    # Output of FC
    out = self.fc_model(cnn_out)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
