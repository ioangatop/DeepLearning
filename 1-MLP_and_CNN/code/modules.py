"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.init_weights = np.random.normal(0, 0.0001, [in_features, out_features])
    self.init_bias = np.zeros(shape=out_features)
    self.init_grad_weights = np.zeros((in_features, out_features))

    self.params = {'weight': self.init_weights, 'bias': self.init_bias}
    self.grads = {'weight': self.init_grad_weights, 'bias': self.init_bias}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # x: batch_size * features
    self.x = x

    out = np.dot(x, self.params['weight']) + self.params['bias']
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # dx = np.dot(self.params['weight'], dout.T)
    dx = np.dot(dout, self.params['weight'].T)
 
    self.grads['weight'] = np.dot(self.x.T, dout)
    self.grads['bias'] = np.sum(dout, axis=0)  # mean or sum ??
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = out = np.maximum(x, 0)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    relu_grad = self.x > 0
    dx = relu_grad*dout
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b = x.max(axis=1)
    y = np.exp(x - b[:, None])
    self.x = out = y / y.sum(axis=1)[:, None]
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = self.x.shape[0]
    n_classes = self.x.shape[1]

    diag = np.zeros((batch_size, n_classes, n_classes))
    temp_diag = np.arange(n_classes)
    diag[:, temp_diag, temp_diag] = self.x

    softmax_der = diag - self.x[:, :, None] * self.x[:, None, :]
    
    dx = np.matmul(dout[:, None, :], softmax_der).squeeze()
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # To avoid numerical issues with logarithm, clip the
    # predictions to [EPSILON, 1 − EPSILON] range.

    EPSILON = 1e-30
    x = np.clip(x, a_min=EPSILON, a_max=None)
    out = np.mean(np.sum(-y * np.log(x), axis=1)) #[:, None]
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    EPSILON = 1e-30
    x = np.clip(x, EPSILON, None)
    batch_size = y.shape[0]

    dx = -(y/x)/batch_size
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
