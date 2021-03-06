import numpy as np

class Softmax(object):

  def __init__(self, X,y,W):
    self.X = X
    self.y = y
    self.W = W

  def softmax_loss_vectorized(self,X,y,reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    W = self.W
    dW = np.zeros_like(W)
    
 

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N, D = X.shape
    scores = np.dot(X,W)
    expTmp = np.exp(scores)
    possTmp = expTmp/(np.sum(expTmp,1).reshape(N,1))
    regularLoss = sum(sum(W**2))

    labelArray = np.zeros_like(possTmp)#label where the possibity should be zero
    tmpList = np.array(range(N))
    labelArray[tmpList,y] = 1


    loss = sum(sum(-np.log(possTmp) * labelArray))/N + reg * regularLoss

    delta_y = possTmp
    delta_y -= labelArray
  
    dW = np.dot(np.transpose(X),delta_y)/N + reg * W * 2
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

  def train(self,X_dev,y_dev,reg,lr,epochNum,batch_size):
    X = self.X
    y = self.y
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    if(iterations_per_epoch == 1):
      batch_size = num_train

    maxAcc = 0.0
    count = 0
    bestW = None
    for k in range(epochNum):
      for i in range(iterations_per_epoch):
        import random
        sequence = range(num_train)
        batchSequence = np.array(random.sample(sequence,batch_size))
        X_batch = self.X[batchSequence]
        y_batch = self.y[batchSequence]

        loss, grad = self.softmax_loss_vectorized(X_batch,y_batch,reg)
	self.W -= lr*grad
      tmpAcc = self.getAcc(X_dev,y_dev)
      count += 1
      if(tmpAcc > maxAcc):
        maxAcc = tmpAcc
        bestW = self.W.copy()
      #if count % 20 == 0:
        #print 'Epoch %d ,Acc: %e' % (count, tmpAcc)
      #print 'Epoch %d ,Acc: %e' % (count, tmpAcc)
    self.W = bestW.copy()
  def predict(self,X):
    W =self.W
    N, D = X.shape
    scores = np.dot(X,W)
    expTmp = np.exp(scores)
    possTmp = expTmp/(np.sum(expTmp,1).reshape(N,1))
  
    maxPoss = np.max(possTmp,1).reshape(N,1)
    y_pred = (np.where(possTmp == maxPoss))[1]
    
    return y_pred

  def getAcc(self,X,y):
    y_pred = self.predict(X)
    Acc = (y_pred == y).mean()
    return Acc
    

  
