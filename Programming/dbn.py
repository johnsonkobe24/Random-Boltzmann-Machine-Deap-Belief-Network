import math
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid


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
        self.h_all=[]
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
        self.h_all.append(self.h_sample)
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
        
class DBN:
    def __init__(self, n_visible, layers, k, lr, max_epochs):
        """ 
        The Deep Belief Network (DBN) class
        Args:
            n_visible: Dimension of visible features layer
            layers: a list, the dimension of each hidden layer, e.g,, [500, 784]
            k: gibbs sampling steps
            lr: learning rate, remains constant through train
            max_epochs: Number of train epochs
        """
        # Instantiate DBN class constants
        #---------------------------------------------
        self.n_visible = n_visible
        self.layers = layers
        self.k = k
        self.lr = lr
        self.max_epochs = max_epochs
        # Instantiate RBM components through the layers
        #----------------------------------------------
        self.rbms = []
        rbm = RBM(n_visible=n_visible, n_hidden=layers[0], k=3, lr=0.01, max_epochs=100)
        self.rbms.append(rbm)
        for i in range(1, len(self.layers)):
            rbm1=RBM(n_visible=500, n_hidden=layers[i], k=3, lr=0.01, max_epochs=100)
            self.rbms.append(rbm1)
            #print("complete")
        self.te_list = np.zeros((len(self.rbms), self.max_epochs))
        self.ve_list = np.zeros((len(self.rbms), self.max_epochs))

    def fit(self, X, valid_X):
        """ The training process of a DBN, basically we train RBMs one by one
        Args:
            X: the train images, numpy matrix
            valid_X: the valid images, numpy matrix
        """

        # zero lists for reconstruction errors
        self.te_list = np.zeros((len(self.rbms), self.max_epochs))
        self.ve_list = np.zeros((len(self.rbms), self.max_epochs))

        # iterate over all RBMs
        for i in range(len(self.rbms)):
            if i > 0:  # get new data
                train = self.rbms[0].gibbs_k(X, k=1)[2]
                valid = self.rbms[0].gibbs_k(valid_X, k=1)[2]

                # iterate over the RBM's h_v and sample_h method
                # to generate the train and valid data.
                #print("complete")
            else:
                train = X
                valid = valid_X

            # iterate over all epochs
            for epoch in range(self.max_epochs):
                shuff = shuffle_corpus(train)
                for x in shuff:
                    # update the RBM weights
                    self.rbms[i].update(x)
                    #print("complete")

                te = self.rbms[i].evaluate(train)
                ve = self.rbms[i].evaluate(valid)
                self.te_list[i][epoch] = te
                self.ve_list[i][epoch] = ve

                # Print optimization trajectory
                train_error = "{:0.4f}".format(te)
                valid_error = "{:0.4f}".format(ve)
                print(f"Epoch {epoch + 1} :: RBM {i + 1} :: \t " +
                      f"Train Error {train_error} :: Valid Error {valid_error}")
            print("\n")


def fit_mnist_dbn(n_visible, layers, k, max_epochs, lr):
    train_data = np.genfromtxt('/Users/johnson/Desktop/S22_HW2_handout/Programming/data/digitstrain.txt', delimiter=",")
    train_X = train_data[:, :-1] 
    train_Y = train_data[:, -1]
    train_X = train_X[-900:]

    valid_data = np.genfromtxt('/Users/johnson/Desktop/S22_HW2_handout/Programming/data/digitsvalid.txt', delimiter=",")
    valid_X = valid_data[:, :-1][-300:]
    valid_Y = valid_data[:, -1]

    test_data = np.genfromtxt('/Users/johnson/Desktop/S22_HW2_handout/Programming/data/digitstest.txt', delimiter=",")
    test_X = test_data[:, :-1][-300:]
    test_Y = test_data[:, -1]

    train_X = binary_data(train_X)
    valid_X = binary_data(valid_X)
    test_X = binary_data(test_X)

    n_visible = train_X.shape[1]
    
    dbn = DBN(n_visible=n_visible, layers=layers, 
              k=k, max_epochs=max_epochs, lr=lr)
    dbn.fit(X=train_X, valid_X=valid_X)
    return dbn

if __name__ == "__main__":
    
    np.seterr(all='raise')
    plt.close('all')

    a=fit_mnist_dbn(n_visible=784, layers=[500, 784], k=1, max_epochs=300, lr=0.01)
#%%
    fig, host = plt.subplots(figsize=(8,5))
    host.set_ylim(0, 11)
    host.set_ylabel("reconstruction error")
    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    p1, = host.plot(a.te_list[0],color=color1, label="layer1_train")
    p2, = host.plot(a.ve_list[0], color=color2, label="layer1_valid")
    lns = [p1, p2]
    host.legend(handles=lns, loc='best')
    plt.show() 
    

#%%
h2=np.random.binomial(1,0.5,(100,784))
prob_v=a.rbms[1].v_h(h2)
h1=a.rbms[1].sample_v(prob_v)
h1=a.rbms[1].gibbs_k(h1,k=20000)[3]
prob_v=a.rbms[0].v_h(h1)
images=a.rbms[0].sample_v(prob_v)
#%%

fig1=plt.figure(1)
columns=10
rows=10
for i in range(1,101):
        fig1.add_subplot(rows, columns, i)
        data=prob_v[i-1].reshape(28,28)
        plt.imshow(data)
        plt.axis('off')

plt.subplot_tool()
plt.show()
