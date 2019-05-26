import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_neurons = n_neurons
    self.eps = eps

    self.gamma = torch.nn.Parameter(torch.ones(n_neurons))
    self.beta = torch.nn.Parameter(torch.zeros(n_neurons))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    assert input.shape[1] == self.n_neurons, "The shape of the input tensor is not correct."

    # compute mean
    mean = input.mean(dim=0)

    # compute variance
    var = input.var(dim=0, unbiased=False)

    # normalize
    input_norm = (input-mean)/(torch.sqrt(var + self.eps))

    # scale and shift
    out = self.gamma*input_norm + self.beta
    ########################
    # END OF YOUR CODE    #
    #######################

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    ####### Forward pass of batch normalization ######

    # In this section, we have to perform the forward pass of batch normalization
    # with more intermediate steps, since we want to propagate error terms. 
    # To illustrate it better, we began from the bottom and follow our way to the top.
    # In that way, we unfolded every function step by step.

    # Step 3.2.3: Calculate variance
    var = input.var(dim=0, unbiased=False)

    # Step 3.2.2: add eps for numerical stability, then sqrt
    sqrt_var = torch.sqrt(var + eps)

    # Step 3.2: ivert sqrtwar
    inv_sqrt_var = 1./sqrt_var

    # Step 3.1.1: Calculate mean
    mean = input.mean(dim=0)

    # Step 3.1: subtract mean vector of every trainings example
    input_mean = input - mean

    # Step 3 - Execute normalization
    input_norm = input_mean * inv_sqrt_var 

    # Step 2: Nor the two transformation steps
    scaled_input_norm = gamma * input_norm

    # Step 1: scale and shift
    out = scaled_input_norm + beta
    #################################################
    # store tensors and non-tensorial constants
    ctx.save_for_backward(gamma, inv_sqrt_var, mean, input)
    ctx.foo = eps
    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Retrieve saved tensors and constants
    gamma, ivar, mean, input = ctx.saved_tensors
    eps = ctx.saved_tensors

    # Check which inputs need gradients
    input_needs_grad, gamma_needs_grad, beta_needs_grad = ctx.needs_input_grad

    # Get the batch size (=N)
    N, _ = grad_output.shape

    # reconstruct the input_norm
    input_norm = (input - mean) * ivar
    grand_input_norm = grad_output * gamma

    ##### Gradient wrt beta #####
    grad_beta = grad_output.sum(dim=0) if beta_needs_grad else None

    #### Gradient wrt gamma ####
    grad_gamma = (input_norm*grad_output).sum(dim=0) if gamma_needs_grad else None
    
    #### Gradient wrt input ####
    term1 = N*grand_input_norm 
    term2 = torch.sum(grand_input_norm, dim=0)
    term3 = input_norm*torch.sum(grand_input_norm*input_norm, dim=0)
    grad_input = (1. / N) * ivar * (term1 - term2 - term3) if input_needs_grad else None

    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_neurons = n_neurons
    self.eps = eps

    self.gamma = torch.nn.Parameter(torch.ones(n_neurons))
    self.beta = torch.nn.Parameter(torch.zeros(n_neurons))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    assert input.shape[1] == self.n_neurons, "The shape of the input tensor is not correct."

    bn_fct = CustomBatchNormManualFunction()
    out = bn_fct.apply(input, self.gamma, self.beta, self.eps)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
