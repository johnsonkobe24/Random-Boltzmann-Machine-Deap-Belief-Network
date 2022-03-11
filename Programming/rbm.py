import os
import sys
import time
import math
import random
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network._multilayer_perceptron import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------
# Utility functions, do not modify
#-----------------------------------------------------------------------
'''
if not os.path.exists('../plot'):
    os.makedirs('../plot')
if not os.path.exists('../dump'):
    os.makedirs('../dump')
'''
seed = 10417617

def binary_data(inp):
    # Do not modify
    return (inp > 0.5) * 1.

def sigmoid(x):
    """
    Args:
        x: input
    Returns: the sigmoid of x
    """
    # Do not modify
    return 1 / (1 + np.exp(-x))

def xavier_init(n_input, n_output):
    """
    # Use Xavier weight initialization
    # Xavier Glorot and Yoshua Bengio, 
    "Understanding the difficulty of training deep feedforward neural networks"
    """
    # Do not modify
    b = np.sqrt(6/(n_input + n_output))
    return np.random.normal(0,b,(n_input, n_output))

def shuffle_corpus(X, y=None):
    """shuffle the corpus randomly
    Args:
        X: the image vectors, [num_images, image_dim]
        y: the image digit, [num_images,], optional
    Returns: The same images and digits (if supplied) with different order
    """ 
    # Do not modify
    random_idx = np.random.permutation(len(X))
    if y is None:
        return X[random_idx]
    return X[random_idx], y[random_idx]

# Do not modify ^^^^
#-----------------------------------------------------------------------
class WarmUpMLPClassifier(MLPClassifier):
    """
    The WarmUpMLPClassifier builds on top of the sklearn MLPClassifier.
    overwriting the MLPClassifier's _init__ and _initialize method
    to include DBN or RBM weights as a warm_start.

    Feel free to modify the hyperparemters in this class: you can change solvers,
    learning_rate, momentum, etc

    Args:
        lr: learning rate, remains constant through train
        max_epochs: Number of train SGD epochs
        hidden_layer_sizes: List with dimension of hidden layers
        W: Weights between visible and hidden layer, shape (n_visible, n_hidden)
        hbias: Bias for the hidden layer, shape (n_hidden, )

    Returns:
        Instantiated class with following parameters

    """

    def __init__(self, lr, max_epochs,
                 hidden_layer_sizes, Ws, hbiases):

        # ---------------------------------------------
        # Instantiate WarmUpMLPClassifier class constants
        self.learning_rate_init = lr
        self.max_iter = max_epochs
        self.hidden_layer_sizes = hidden_layer_sizes

        if Ws is not None and hbiases is not None:
            if not isinstance(Ws, list) and not isinstance(Ws, tuple):
                print(f"Error: input Ws needs to be a list or tuple.")
                return
            if not isinstance(hbiases, list) and not isinstance(hbiases, tuple):
                print(f"Error: input hbiases needs to be a list or tuple.")
                return

            assert len(hbiases) == len(Ws), f"Length of hbiases and Ws need to match."

            for i in range(len(hidden_layer_sizes)):
                hidden_layer_size = hidden_layer_sizes[i]
                assert hidden_layer_size == Ws[i].shape[0], f'{i}th layer W size mismatch'
                assert hidden_layer_size == hbiases[i].shape[0], f'{i}th layer bias size mismatch'

            self.hbiases = hbiases
            self.Ws = Ws
        else:
            self.hbiases = None
            self.Ws = None

        # ---------------------------------------------
        self.batch_size = 1
        self.random_state = 1
        self.loss = "log_loss"
        self.activation = "logistic"
        self.solver = "sgd"
        self.learning_rate = "constant"
        self.power_t = 0.5

        self.shuffle = True
        self.tol = 1e-4
        self.verbose = True
        self.warm_start = False

        # Unused hyperparameters, needed for MLPClassifier
        self.max_fun = 15000
        self.alpha = 0.0001
        self.momentum = 0.9
        self.nesterovs_momentum = True
        self.early_stopping = False
        self.validation_fraction = 0.0
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.n_iter_no_change = 10
        # Do not modify  ^^^^
        # ---------------------------------------------

    def _initialize(self, y, layer_units):
        #-------------------------------------------------------
        _STOCHASTIC_SOLVERS = ['sgd', 'adam']
        # set all attributes, allocate weights etc for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0
        self.n_outputs_ = y.shape[1]

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Output for binary class and multi-label
        self.out_activation_ = "softmax"

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(
                layer_units[i], layer_units[i + 1]
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)
        
        # Do not modify ^^^^
        #-------------------------------------------------------


        # Modify/Overwrite coefs_ and intercepts_
        # initialization of layers with RBM/DBN layer weights
        print("Complete")
        if self.Ws is not None:
            self.coefs_[0]=np.array(self.Ws).squeeze(axis=0).transpose(1,0)
            self.intercepts_[0]=np.array(self.hbiases).squeeze(axis=0)
            print("Complete")


        #-------------------------------------------------------
        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
            else:
                self.best_loss_ = np.inf
        # Do not modify ^^^^
        #-------------------------------------------------------
class RBM:
    def __init__(self, n_visible, n_hidden, k, lr, max_epochs):
        """The RBM base class
        Args:
            n_visible: Dimension of visible features layer
            n_hidden: Dimension of hidden layer
            k: gibbs sampling steps
            lr: learning rate, remains constant through train
            max_epochs: Number of train epochs
        Returns:
            Instantiated class with following parameters
            hbias: Bias for the hidden layer, shape (n_hidden, )
            vbias: Bias for the visible layer, shape (n_visible, )
            W: Weights between visible and hidden layer, shape (n_visible, n_hidden)
        """
        # Instantiate RBM class constants
        #---------------------------------------------
        self.n_visible = n_visible
        self.n_hidden =n_hidden
        self.k = k
        self.lr = lr
        self.max_epochs = max_epochs
        self.v_all=[]
        # Initialize hidden and visible biases with zeros
        # Initialize visible weights with Xavier (random_weight_init above)
        # Initialize classification weights with Xavier
        #---------------------------------------------
        self.hbias = np.zeros(n_hidden)
        self.vbias = np.zeros(n_visible)
        self.W = xavier_init(self.n_hidden, self.n_visible) 
        self.v_sample=np.zeros(n_visible)
        self.h_sample=np.zeros(n_hidden)

    def h_v(self, v):
        """ Transform the visible vector to hidden vector and 
            compute its probability being 1
        Args:
            v: Visible vector (n_visible, )
        Returns:
            1. Probability of hidden vector h being 1 p(h=1|v), shape (n_hidden, )
        """
        return sigmoid(np.matmul(v,self.W.T)+self.hbias)

    def sample_h(self, h_prob):
        """ 
        Sample a hidden vector given the distribution p(h=1|v)
        
        Args: 
            h_prob: probability vector p(h=1|v), shape (n_hidden, )
        Return:
            1. Sampled hidden vectors, shape (n_hidden, )
        """
        #np.random.seed(seed)
        sampled_prob_h_v=np.random.binomial(1,h_prob )
        sampled_h=binary_data(sampled_prob_h_v)
        return sampled_h 

    def v_h(self, h):
        """
        Transform the hidden vector to visible vector and
            compute its probability being 1
        
        Args:
            h: the hidden vector h (n_hidden,)
        Return:
            Hint: sigmoid provided function.
            1. Probability of output visible vector v being 1, shape (n_visible,)
        """
        return sigmoid(np.matmul(h,self.W) + self.vbias)

    def sample_v(self, v_prob, v_true=None, v_observation=None):
        """ 
        Sample a visible vector given the distribution p(v=1|h)
        Args: 
            v_prob: probability vector p(v=1|h), shape (n_visible,)
            v_true: Ground truth vector v, shape (n_visible, )
            v_observation: a 0-1 mask that tells which index is observed by the RBM, 
                    where 1 means observed, and 0 means not observed, shape (n_visible, )
            
            Example:
            Say v is of size (2,), v_true is [1, 0], and the v_observation is [0, 1], 
            then we reveal the second true entry "0" to the RBM.
            
            When you do gibbs sampling, you "inject" the observed part of v_true: 
                      v_true * v_observation
            to the RBM, so that you have a super certain probability distribution,
            on the observed indexes. Here the "*" is entry-wise multiplication.                        
        Return:
            Hint: NumPy binomial sample
            1. Sampled visible vector, binary in our experiment
                shape (n_visible,)
        """
        #np.random.seed(seed)
        sampled_prob_v_h=np.random.binomial(1,v_prob)
        sampled_v=binary_data(sampled_prob_v_h)
        if isinstance(v_observation, np.ndarray)==True:
            for i in range(len(v_observation)):
                if v_observation[i]==1:
                    if v_true[i]==1:
                        sampled_v[i]=1
                    else:
                        sampled_v[i]=0
        return sampled_v

    def gibbs_k(self, v, k=0, v_true=None, v_observation=None):
        """ 
        The contrastive divergence k (CD-k) procedure,        
        with the possibility of injecting v_true observed values.
        Args:
            v: the input visible vector (n_visible,)
            v_true: Ground truth vector v, shape (n_visible, )
            v_observation: a 0-1 mask that tells which index is observed by the RBM, 
                    where 1 means observed, and 0 means not observed, shape (n_visible, )
            k: the number of gibbs sampling steps, scalar (int)
        Return:
            Hint: complete the tests and use the methods h_v, sample_h, v_h, sample_v
            1. h0: Hidden vector sample with one iteration (n_hidden,)
            2. v0: Input v (n_visible,)
            3. h_sample: Hidden vector sample with k iterations  (n_hidden,)
            4. v_sample: Visible vector sampled wit k iterations (n_visible,)
            5. h_prob: Prob of hidden being 1 after k iterations (n_hidden,)
            6. v_prob: Prob of visible being 1 after k itersions (n_visible,)
        """
        v0 = binary_data(v)
        h_prob = self.h_v(v0)
        h0=self.sample_h(h_prob)
        h_sample=h0
        # complete
        
        for i in range (k if k>0 else self.k):
            v_prob=self.v_h(h_sample)
            v_sample=self.sample_v(v_prob,v_true, v_observation)
            h_prob=self.h_v(v_sample)
            h_sample=self.sample_h(h_prob)
            #print("complete")


        return h0, v0, h_sample, v_sample, h_prob, v_prob

    def update(self, x):
        """ 
        Update the RBM with input v.
        Args:
            v: the input data X , shape (n_visible,)
        Return: self with updated weights and biases
            Hint: Compute all the gradients before updating weights and biases.
        """
        self.x=x
        h0, v0, h_sample, v_sample, h_prob, v_prob = self.gibbs_k(x)
        self.v_sample=v_sample
        self.h_sample=h_sample
        h_prob_x=self.h_v(x)           
        self.W = self.W + self.lr*(np.outer(h_prob_x, x)-np.outer(h_prob, v_sample)) 
        self.hbias = self.hbias + self.lr* (h_prob_x-h_prob)
        self.vbias = self.vbias + self.lr* (x-v_sample)
        self.v_all.append(self.v_sample)
        # complete


    def evaluate(self, X, k=0):
        """ 
        Compute reconstruction error
        Args:
            X: the input X, shape (len(X), n_visible)
        Return:
            The reconstruction error, shape a scalar
        """
        error=0
        sample=self.gibbs_k(X)[3]
        for i in range(len(X)):
            error=error+np.sqrt(np.sum((X[i]-sample[i])**2))
        return error/len(X)

    def fit(self, X, valid_X):
        """ 
        Fit RBM, do not modify. Note that you should not use this function for conditional generation.
        Args:
            X: the input X, shape (len(X), n_visible)
            X_valid: the validation X, shape (len(valid_X), n_visible)
        Return: self with trained weights and biases
        """
        # Do not modify
        # Initialize trajectories
        self.loss_curve_train_ = []
        self.loss_curve_valid_ = []
        # Train
        for epoch in range(self.max_epochs):
            shuffled_X = shuffle_corpus(X)
            
            for i in range(len(shuffled_X)):
                x = shuffled_X[i]
                self.update(x)

            # Evaluate
            train_recon_err = self.evaluate(shuffled_X)
            valid_recon_err = self.evaluate(valid_X)
            self.loss_curve_train_.append(train_recon_err)
            self.loss_curve_valid_.append(valid_recon_err)
            
            # Print optimization trajectory
            train_error = "{:0.4f}".format(train_recon_err)
            valid_error = "{:0.4f}".format(valid_recon_err)
            print(f"Epoch {epoch+1} :: \t Train Error {train_error} \
                  :: Valid Error {valid_error}")

            print(epoch)
        print("\n\n")
#%%
if __name__ == "__main__":

    np.seterr(all='raise')

    parser = argparse.ArgumentParser(description='data, parameters, etc.')
    parser.add_argument('-train', type=str, help='training file path', default='/Users/johnson/Desktop/S22_HW2_handout/Programming/data/digitstrain.txt')
    parser.add_argument('-valid', type=str, help='validation file path', default='/Users/johnson/Desktop/S22_HW2_handout/Programming/data/digitsvalid.txt')
    parser.add_argument('-test', type=str, help="test file path", default="/Users/johnson/Desktop/S22_HW2_handout/Programming/data/digitstest.txt")
    parser.add_argument('-max_epochs', type=int, help="maximum epochs", default=100)

    parser.add_argument('-n_hidden', type=int, help="num of hidden units", default=25)
    parser.add_argument('-k', type=int, help="CD-k sampling", default=1)
    parser.add_argument('-lr', type=float, help="learning rate", default=0.01)
    parser.add_argument('-minibatch_size', type=int, help="minibatch_size", default=50)

    args = parser.parse_args()

    train_data = np.genfromtxt(args.train, delimiter=",")
    train_X = train_data[:, :-1] 
    train_Y = train_data[:, -1]
    train_X = binary_data(train_X)

    valid_data = np.genfromtxt(args.valid, delimiter=",")
    valid_X = valid_data[:, :-1]
    valid_X = binary_data(valid_X)
    valid_Y = valid_data[:, -1]

    test_data = np.genfromtxt(args.test, delimiter=",")
    test_X = test_data[:, :-1]
    test_X = binary_data(test_X)
    test_Y = test_data[:, -1]

    n_visible = train_X.shape[1]

    print("input dimension is " + str(n_visible)) 
    rbm = RBM(n_visible=n_visible, n_hidden=args.n_hidden, 
              k=args.k, lr=args.lr, max_epochs=args.max_epochs)
    rbm.fit(X=train_X, valid_X=valid_X)

#%%
    fig, host = plt.subplots(figsize=(8,5))
    host.set_ylim(0, 9)
    host.set_ylabel("loss")
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    p1, = host.plot(rbm.loss_curve_train_,color=color1, label="train_loss")
    p2, = host.plot(rbm.loss_curve_valid_, color=color2, label="validation_loss")
    lns = [p1, p2]
    host.legend(handles=lns, loc='best')
    plt.show()
#%%
    rbm1 = RBM(n_visible=n_visible, n_hidden=100, k=args.k, lr=args.lr, max_epochs=100)
    rbm1.fit(X=train_X, valid_X=valid_X)
#%%
    rbm_clf = WarmUpMLPClassifier(lr=0.01, max_epochs=3, hidden_layer_sizes=(1000,),Ws=[rbm.W,], hbiases=[rbm.hbias,])
    rbm_clf.fit(train_X, train_Y)
    rbm_clf1 = WarmUpMLPClassifier(lr=0.001, max_epochs=10, hidden_layer_sizes=(1000,),Ws=[xavier_init(args.n_hidden,n_visible),], hbiases=[np.zeros(1000),])
    rbm_clf1.fit(train_X, train_Y)
    rbm_clf2 = WarmUpMLPClassifier(lr=0.01, max_epochs=50, hidden_layer_sizes=(100,),Ws=[rbm1.W,], hbiases=[rbm1.hbias,])
    rbm_clf2.fit(train_X, train_Y)
    rbm_clf3 = WarmUpMLPClassifier(lr=0.01, max_epochs=50, hidden_layer_sizes=(100,),Ws=[xavier_init(100,784),], hbiases=[np.zeros(100),])
    rbm_clf3.fit(train_X, train_Y)
    # you can access the train and validation error trajectories
    # from the self.loss_curve_train_ and self.loss_curve_valid_ attributes
    fig, host = plt.subplots(figsize=(8,5))
    host.set_ylim(0, 3)
    host.set_ylabel("loss")
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    p1, = host.plot(rbm_clf2.loss_curve_,color=color1, label="MLP_RBM")
    p2, = host.plot(rbm_clf3.loss_curve_, color=color2, label="MLP_random")
    lns = [p1, p2]
    host.legend(handles=lns, loc='best')
    plt.show() 
#%%
fig1=plt.figure(1)
columns=10
rows=10
for i in range(1,101):
        fig1.add_subplot(rows, columns, i)
        data=rbm.W[i-1].reshape(28,28)
        data
        plt.imshow(data)
        plt.axis('off')

plt.subplot_tool()
plt.show() 
#%%  
num_test = 10
mask = np.zeros((28, 28))
mask[0:14] = 1
mask_1d = mask.reshape(-1)
masked_X=[]
for i in range(num_test):
    masked_X.append(train_X[i*300])
masked_X=np.array(masked_X)
v_sample=np.random.binomial(1,0.5,784)
images0=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[0], v_observation=mask_1d)[3]
images1=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[1], v_observation=mask_1d)[3]
images2=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[2], v_observation=mask_1d)[3]
images3=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[3], v_observation=mask_1d)[3]
images4=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[4], v_observation=mask_1d)[3]
images5=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[5], v_observation=mask_1d)[3]
images6=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[6], v_observation=mask_1d)[3]
images7=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[7], v_observation=mask_1d)[3]
images8=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[8], v_observation=mask_1d)[3]
images9=rbm.gibbs_k(v_sample,k=20000,v_true=masked_X[9], v_observation=mask_1d)[3]
images=np.stack([images0,images1,images2,images3,images4,images5,images6,images7,images8,images9])
images.shape

#%%
fig1=plt.figure(1)
columns=10
rows=1
for i in range(1,11):
        fig1.add_subplot(rows, columns, i)
        data=images[i-1].reshape(28,28)
        data
        plt.imshow(data)
        plt.axis('off')

plt.subplot_tool()
plt.show()   
#%%
v_sample=np.random.binomial(1,0.5,(10,784))
image=rbm.gibbs_k(v_sample, k=20000)
images=image[5]
#%%
num_test = 10
mask = np.zeros((28, 28))
mask[0:14] = 1
mask_1d = mask.reshape(-1)
masked_X=[]
for i in range(num_test):
    masked_X.append(train_X[i*300])
images=[(x*mask_1d).reshape(28,28) for x in masked_X]


