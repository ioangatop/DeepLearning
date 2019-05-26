"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object.
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    if n_hidden != []:
      n_hidden.insert(0, n_inputs)
      n_hidden.append(n_classes)

      self.layer = []
      for idx_layer in range(len(n_hidden)-1):
        self.layer.append(LinearModule(n_hidden[idx_layer], n_hidden[idx_layer+1]))

    else:
      self.layer = []
      self.layer.append(LinearModule(n_inputs, n_classes))

    self.relu = ReLUModule()
    self.softmax = SoftMaxModule()
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
    for idx_layer in range(len(self.layer) - 1):
      x = self.relu.forward(self.layer[idx_layer].forward(x))
    out = self.softmax.forward(self.layer[-1].forward(x))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dout = self.layer[-1].backward(self.softmax.backward(dout))

    for idx_layer in range(len(self.layer)-2, -1, -1):
      dout = self.layer[idx_layer].backward(self.relu.backward(dout))
      
    ########################
    # END OF YOUR CODE    #
    #######################

    return
