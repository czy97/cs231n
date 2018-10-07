import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1'] = np.zeros((num_filters,input_dim[0],filter_size,filter_size))
    for f_val in range(num_filters):
      for c_val in range(input_dim[0]):
        self.params['W1'][f_val][c_val] = weight_scale * np.random.randn(filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)

    
    
    self.params['W2'] = weight_scale * np.random.randn(num_filters * input_dim[1]/2 * input_dim[1]/2, hidden_dim)
    
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #print X.shape
    out_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    #print out_conv.shape
    N_conv, F_conv, H_conv, W_conv = out_conv.shape
    
    out_conv_vector = out_conv.reshape(N_conv,F_conv*H_conv*W_conv)
    #print out_conv_vector.shape
    
    out2, cache2 = affine_relu_forward(out_conv_vector, W2, b2)
    scores, cache3 = affine_forward(out2, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    N = X.shape[0]
    expTmp = np.exp(scores)
    possTmp = expTmp/(np.sum(expTmp,1).reshape(N,1))
    
    
    regularLoss = np.sum(self.params['W1']**2,(0,1,2,3)) + np.sum(self.params['W2']**2,(0,1)) + np.sum(self.params['W3']**2,(0,1))
    #regularLoss = np.sum(self.params['W2']**2,(0,1)) + np.sum(self.params['W3']**2,(0,1))
    labelArray = np.zeros_like(possTmp)#label where the possibity should be zero
    tmpList = np.array(range(N))
    labelArray[tmpList,y] = 1
    loss = sum(sum(-np.log(possTmp) * labelArray))/N + self.reg * regularLoss *0.5

    dout = possTmp.copy()
    dout -= labelArray
    
    dout2, grads['W3'], grads['b3'] = affine_backward(dout, cache3)
    grads['W3'] = grads['W3']/N + self.reg * self.params['W3']
    grads['b3'] = grads['b3']/N

    dout1, grads['W2'], grads['b2'] = affine_relu_backward(dout2, cache2)
    grads['W2'] = grads['W2']/N + self.reg * self.params['W2']
    grads['b2'] = grads['b2']/N
    
    dout1 = dout1.reshape(N_conv, F_conv, H_conv, W_conv)

    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout1, cache_conv)
    grads['W1'] = grads['W1']/N + self.reg * self.params['W1']
    #grads['W1'] = grads['W1']/N
    grads['b1'] = grads['b1']/N
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
