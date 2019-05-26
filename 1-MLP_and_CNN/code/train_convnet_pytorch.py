"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn
from torch import optim
import datetime

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 22000 #4*5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  # Get the top probability and top class for each image
  top_p, top_class = predictions.topk(1, dim=1)

  # Check if the predicted classes match the labels
  equals = top_class == targets.view(*top_class.shape)

  # Calculate the percentage of correct predictions
  accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
  
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  ################### GPU CUDA CHECK ####################
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  # check if CUDA is available
  if not torch.cuda.is_available():
      print('CUDA is not available.  Training on CPU ... \n')
  else:
      print('CUDA is available!  Training on GPU ... \n')
  #######################################################

  ######### DEFINITION OF MODEL AND PARAMETERS ##########
  # Get Data
  #   Since the "nn.CrossEntropyLoss()" or "nn.NLLLoss()" does not  support
  #   one-hot-vector, we load the data in dense format.
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, one_hot=False, validation_size=0)

  # Get a batch to find the #channels and #classes of the data
  x, t = cifar10["train"].next_batch(FLAGS.batch_size)
  INPUT_DIM = x.shape[1]
  OUTPUT_DIM = 10

  # Define the Neural Network
  model = nn.DataParallel(ConvNet(INPUT_DIM, OUTPUT_DIM).to(device))

  # Define the loss
  criterion = nn.CrossEntropyLoss()  # criterion = nn.NLLLoss()

  # Define Optimizer
  optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
  ######################################################
  
  ############ Data transformation #############
  # Get all the data from the test set
  x_test, t_test = cifar10['test'].images, cifar10['test'].labels

  # Convert numpy.ndarray to touch.Tensor
  x_test = torch.from_numpy(x_test).to(device)
  t_test = torch.from_numpy(t_test).long().to(device)
  ##############################################
  train_loss, test_loss, train_accuracy, test_accuracy = [], [], [], []
  eval_steps = []
  ######### TRAIN OF THE MODEL ##########
  for step in range(FLAGS.max_steps):

    # Get batch data
    x, t = cifar10["train"].next_batch(FLAGS.batch_size)
    
    # Convert numpy.ndarray to touch.Tensor
    x = torch.from_numpy(x).to(device)
    t = torch.from_numpy(t).long().to(device)

    # Forward pass
    y = model(x)

    # Calculate the loss
    loss = criterion(y, t)

    # Gradients for the parameters are calculated
    loss.backward()

    # Update weights
    optimizer.step()

    # clear the gradients of all optimized variables
    optimizer.zero_grad()

    # Store train accuracy and loss
    train_loss.append(loss.item())
    train_accuracy.append(accuracy(y, t))
    ##############################################

    if ((step % FLAGS.eval_freq) == 0) or (step == FLAGS.max_steps - 1):
      ################# NOTES ###################
      # - model.eval() will notify all your layers that you are in eval mode, 
      #   that way, batchnorm or dropout layers will work in eval model instead of training mode.
      # - torch.no_grad() impacts the autograd engine and deactivate it. 
      #   It will reduce memory usage and speed up computations but you won’t be able to backprop.
      ###########################################
      eval_steps.append(step)
      ############## Evaluate model #############
      with torch.no_grad():
        
        # model in evaluation mode
        model.eval()

        # Forward pass
        y_test = model(x_test)

        # Loss
        test_loss.append(criterion(y_test, t_test).item())

        # Measure Accuracy
        test_accuracy.append(accuracy(y_test, t_test))

        print("Step {!s:5}: Current Train Accuracy: {:.3f}      Test Accuracy: {:.3f}".format(step, train_accuracy[-1], test_accuracy[-1]))
        print("            Current Train Loss: {:.3f}          Test Loss: {:.3f}\n".format(train_loss[-1], test_loss[-1], '.3f'))

      ###########################################

      # model back to train mode
      model.train()

  ######################################################

  print("\nTraining Finised")
  # Save Train and Test accuracies and losses
  time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  file_name = 'covnet_pytorch_' + time + '.npz'

  np.savez(file_name, eval_steps = eval_steps, train_loss=train_loss, test_loss=test_loss, train_accuracy=train_accuracy, test_accuracy=test_accuracy)

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))
  print("\n")

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()