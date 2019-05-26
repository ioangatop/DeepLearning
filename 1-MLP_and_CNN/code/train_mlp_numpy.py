"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100' #100
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
  predicted_class = np.argmax(predictions, axis=1)[:, None]

  # Check if the predicted classes match the labels
  equals = predicted_class == targets

  # Calculate the percentage of correct predictions
  accuracy = np.mean(equals.astype(float))

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  ################### Initialize Model #####################
  # Get batch to initialize the model
  cifar10 = cifar10_utils.get_cifar10(data_dir=FLAGS.data_dir, one_hot=True, validation_size=0)
  x, t = cifar10["train"].next_batch(FLAGS.batch_size)

  # Initialize the model
  _, INPUT_DIM = x.reshape(FLAGS.batch_size, -1).shape
  _, OUTPUT_DIM = t.shape

  model = MLP(INPUT_DIM, dnn_hidden_units, OUTPUT_DIM)

  # loss function
  criterion = CrossEntropyModule()
  ##########################################################

  ################### Configurate Data #####################
  # Get all the data both from train and test set
  x_test, t_test = cifar10['test'].images, cifar10['test'].labels

  # from one-hot-encode to integer representation
  t_test = np.argmax(t_test, axis=1)[:, None]

  # Flatten the input ndarray
  x_test = x_test.reshape(x_test.shape[0], -1)

  ##########################################################
  train_loss, test_loss, train_accuracy, test_accuracy = [], [], [], []
  eval_steps = []
  ######### TRAIN OF THE MODEL ##########
  for step in range(FLAGS.max_steps):

    x, t = cifar10["train"].next_batch(FLAGS.batch_size)

    # flatten
    x = x.reshape(x.shape[0], -1)

    # forward
    y = model.forward(x)

    # calculate loss
    loss = criterion.forward(y, t)

    # backward
    dout = criterion.backward(y, t)
    model.backward(dout) 

    # update weights
    for layer in model.layer:
      layer.params['weight'] -= FLAGS.learning_rate * layer.grads['weight'] #/FLAGS.batch_size
      layer.params['bias'] -= FLAGS.learning_rate * layer.grads['bias'] #/FLAGS.batch_size

    # Store train accuracy and loss
    train_loss.append(loss)

    t = np.argmax(t, axis=1)[:, None]
    train_accuracy.append(accuracy(y, t))

    ##############################################

    if ((step % FLAGS.eval_freq) == 0) or (step == FLAGS.max_steps - 1):

      ############## Evaluate model #############
      eval_steps.append(step)

      # Forward pass
      y_test = model.forward(x_test)

      # Calculate the Loss
      test_loss.append(criterion.forward(y_test, cifar10['test'].labels))

      # Measure Accuracy
      test_accuracy.append(accuracy(y_test, t_test))

      print("Step {!s:5}: Current Train Accuracy: {:.3f}      Test Accuracy: {:.3f}".format(step, train_accuracy[-1], test_accuracy[-1]))
      print("            Current Train Loss: {:.3f}          Test Loss: {:.3f}\n".format(train_loss[-1], test_loss[-1]))
    ###########################################

  print("\nTraining Finised")
  # Save Train and Test accuracies and losses
  np.savez('mlp_numpy.npz', eval_steps = eval_steps, train_loss=train_loss, test_loss=test_loss, train_accuracy=train_accuracy, test_accuracy=test_accuracy)

  ######################################################

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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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