import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  diff = np.zeros([X.shape[0],W.shape[1]])
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    grad=0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        grad += margin 
        loss += margin 
        dW[:,j] += X[i]
        dW[:,y[i]] += -X[i]    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.  
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  Reg=2*reg*W
  Reg[0,:]=0
  dW /= num_train
  dW += Reg
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  loss = 0.0
  N = X.shape[0]    
  scores = np.array(X.dot(W),np.dtype('f8'))
  correct_scores = scores[np.arange(N),y]
  diff_scores = scores-correct_scores[:,None]+1
  diff_scores[diff_scores<0]=0  
  loss = np.sum(np.sum(diff_scores,axis=1)-1,axis=0)
  diff_scores[diff_scores>0]=1
  diff_scores[np.arange(N),y]=-(np.sum(diff_scores,axis=1)-1)
  loss /= N
  loss += reg * np.sum(W * W)
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dW = np.zeros(W.shape,np.dtype('f8'))# initialize the gradient as zero
  dW=(X.transpose().dot(diff_scores))  
  Reg=2*reg*W
  Reg[0,:]=0
  dW /= N 
  dW += Reg  
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
