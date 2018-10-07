import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class Convnet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=[3, 32, 32], num_filters=[32,32,32], filter_size=7,
               hidden_dim=[100,100,100], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32,use_batchnorm=False):
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
    self.use_batchnorm = use_batchnorm
    self.filter_size = filter_size
   

    self.num_conv_layers = len(num_filters)
    self.a_layers = len(hidden_dim) + 1
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
    
    #init conv layers
    num_filters.insert(0,input_dim[0])
    for i in range(self.num_conv_layers):
      w_name = 'conv_W' + str(i+1)
      b_name = 'conv_b' + str(i+1)
      self.params[w_name] = np.zeros((num_filters[i+1],num_filters[i],filter_size,filter_size))
      for f_val in range(num_filters[i+1]):
        for c_val in range(num_filters[i]):
          self.params[w_name][f_val][c_val] = weight_scale * np.random.randn(filter_size, filter_size)  
      self.params[b_name] = np.zeros(num_filters[i+1])

    if(self.use_batchnorm):
      for i in range(self.num_conv_layers):
        gamma_name = 'conv_gamma' + str(i+1)
        beta_name = 'conv_beta' + str(i+1)
        self.params[gamma_name] = np.ones(num_filters[i+1])
        self.params[beta_name] = np.zeros(num_filters[i+1])


    #self.params['W1'] = np.zeros((num_filters,input_dim[0],filter_size,filter_size))
    #for f_val in range(num_filters):
      #for c_val in range(input_dim[0]):
        #self.params['W1'][f_val][c_val] = weight_scale * np.random.randn(filter_size, filter_size)
    #self.params['b1'] = np.zeros(num_filters)

    #init affine layers
    hidden_dim.insert(0,num_filters[self.num_conv_layers] * input_dim[1]/(2**self.num_conv_layers) * input_dim[1]/(2**self.num_conv_layers))
    hidden_dim.append(num_classes)
    for i in range(self.a_layers):
      W_name = 'a_W' + str(i+1)
      b_name = 'a_b' + str(i+1)
      self.params[W_name] = weight_scale * np.random.randn(hidden_dim[i], hidden_dim[i+1])
      self.params[b_name] = np.zeros(hidden_dim[i+1])
 
    if self.use_batchnorm:
      for i in range(self.a_layers - 1):
        gamma_name = 'a_gamma' + str(i+1)
        beta_name = 'a_beta' + str(i+1)
        self.params[gamma_name] = np.ones(hidden_dim[i+1])
        self.params[beta_name] = np.zeros(hidden_dim[i+1])

    #print self.params.keys()
    #self.params['W2'] = weight_scale * np.random.randn(num_filters * input_dim[1]/2 * input_dim[1]/2, hidden_dim)
    #self.params['b2'] = np.zeros(hidden_dim)
    #self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    #self.params['b3'] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.a_bn_params = []
    if self.use_batchnorm:
      self.a_bn_params = [{'mode': 'train'} for i in xrange(self.a_layers - 1)]

    self.conv_bn_params = []
    if self.use_batchnorm:
      self.conv_bn_params = [{'mode': 'train'} for i in xrange(self.num_conv_layers)]


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
    
    if self.use_batchnorm:
      for bn_param in self.conv_bn_params:
        bn_param[mode] = mode
      for bn_param in self.a_bn_params:
        bn_param[mode] = mode
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.filter_size
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################

    ######################old code#######################
    #print X.shape
    #out_conv, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #print out_conv.shape
    #N_conv, F_conv, H_conv, W_conv = out_conv.shape
    
    #out_conv_vector = out_conv.reshape(N_conv,F_conv*H_conv*W_conv)
    #print out_conv_vector.shape
    
    #out2, cache2 = affine_relu_forward(out_conv_vector, W2, b2)
    #scores, cache3 = affine_forward(out2, W3, b3)
    ####################################################

    #conv layers
    cache_conv = {}
    cache_conv_bn = {}
    cache_a = {}
    cache_a_bn = {}
    outTmp = X
    for i in range(self.num_conv_layers):
      w_name = 'conv_W' + str(i+1)
      b_name = 'conv_b' + str(i+1)
      #print w_name
      #print self.params.keys()
      outTmp, cache_conv[i+1] = conv_relu_pool_forward(outTmp, self.params[w_name], self.params[b_name], conv_param, pool_param)
      if self.use_batchnorm:
        gamma_name = 'conv_gamma' + str(i+1)
        beta_name = 'conv_beta' + str(i+1)
        outTmp, cache_conv_bn[i+1] = spatial_batchnorm_forward(outTmp, self.params[gamma_name], self.params[beta_name], self.conv_bn_params[i])
    
    #affine layers
    N_conv, F_conv, H_conv, W_conv = outTmp.shape
    outTmp = outTmp.reshape(N_conv,F_conv*H_conv*W_conv)  
    #print outTmp.shape
    
    for i in range(self.a_layers - 1):
      W_name = 'a_W' + str(i+1)
      b_name = 'a_b' + str(i+1)
      #print self.params[W_name].shape
      outTmp, cache_a[i+1] = affine_relu_forward(outTmp, self.params[W_name], self.params[b_name])
      if self.use_batchnorm:
        gamma_name = 'a_gamma' + str(i+1)
        beta_name = 'a_beta' + str(i+1)
        outTmp, cache_a_bn[i+1] = batchnorm_forward(outTmp, self.params[gamma_name], self.params[beta_name], self.a_bn_params[i])
    
    W_name = 'a_W' + str(self.a_layers)
    b_name = 'a_b' + str(self.a_layers)    
    scores, cache_out = affine_forward(outTmp, self.params[W_name], self.params[b_name])
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
    
    regularLoss = 0.0
    for i in range(self.num_conv_layers):
      w_name = 'conv_W' + str(i+1)
      regularLoss += np.sum(self.params[w_name]**2,(0,1,2,3))
    for i in range(self.a_layers):
      W_name = 'a_W' + str(i+1)
      regularLoss += np.sum(self.params[W_name]**2,(0,1))
      
    #regularLoss = np.sum(self.params['W1']**2,(0,1,2,3)) + np.sum(self.params['W2']**2,(0,1)) + np.sum(self.params['W3']**2,(0,1))
    #regularLoss = np.sum(self.params['W2']**2,(0,1)) + np.sum(self.params['W3']**2,(0,1))
    labelArray = np.zeros_like(possTmp)#label where the possibity should be zero
    tmpList = np.array(range(N))
    labelArray[tmpList,y] = 1
    loss = sum(sum(-np.log(possTmp) * labelArray))/N + self.reg * regularLoss *0.5

    dout = possTmp.copy()
    dout -= labelArray
    
    W_name = 'a_W' + str(self.a_layers)
    b_name = 'a_b' + str(self.a_layers)    
 
    dout_tmp,grads[W_name], grads[b_name] = affine_backward(dout, cache_out)
    grads[W_name] = grads[W_name]/N + self.reg * self.params[W_name]
    grads[b_name] = grads[b_name]/N
    
    #affine backward
    for i in range(self.a_layers-1,0,-1):
      W_name = 'a_W' + str(i)
      b_name = 'a_b' + str(i)
      if self.use_batchnorm:
        gamma_name = 'a_gamma' + str(i)
        beta_name = 'a_beta' + str(i)
        dout_tmp, dgamma, dbeta = batchnorm_backward(dout_tmp, cache_a_bn[i])
        grads[gamma_name] = dgamma.copy() 
        grads[beta_name] = dbeta.copy()
      dout_tmp, grads[W_name], grads[b_name] = affine_relu_backward(dout_tmp, cache_a[i])
      grads[W_name] = grads[W_name]/N + self.reg * self.params[W_name]
      grads[b_name] = grads[b_name]/N

    #conv backward
    dout_tmp = dout_tmp.reshape(N_conv, F_conv, H_conv, W_conv)
    #print self.num_conv_layers
    for i in range(self.num_conv_layers,0,-1):
      w_name = 'conv_W' + str(i)
      b_name = 'conv_b' + str(i)
      if self.use_batchnorm:
        gamma_name = 'conv_gamma' + str(i)
        beta_name = 'conv_beta' + str(i)
        dout_tmp, dgamma, dbeta = spatial_batchnorm_backward(dout_tmp, cache_conv_bn[i])
        grads[gamma_name] = dgamma.copy() 
        grads[beta_name] = dbeta.copy()
      dout_tmp, grads[w_name], grads[b_name] = conv_relu_pool_backward(dout_tmp, cache_conv[i])
      grads[w_name] = grads[w_name]/N + self.reg * self.params[w_name]
      grads[b_name] = grads[b_name]/N

   
    #######################old code#######################
    #dout2, grads['W3'], grads['b3'] = affine_backward(dout, cache3)
    #grads['W3'] = grads['W3']/N + self.reg * self.params['W3']
    #grads['b3'] = grads['b3']/N

    #dout1, grads['W2'], grads['b2'] = affine_relu_backward(dout2, cache2)
    #grads['W2'] = grads['W2']/N + self.reg * self.params['W2']
    #grads['b2'] = grads['b2']/N
    
    #dout1 = dout1.reshape(N_conv, F_conv, H_conv, W_conv)

    #dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout1, cache_conv)
    #grads['W1'] = grads['W1']/N + self.reg * self.params['W1']
    #grads['b1'] = grads['b1']/N
    #####################################################
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
