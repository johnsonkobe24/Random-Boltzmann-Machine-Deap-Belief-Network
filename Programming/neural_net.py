import argparse

import numpy as np
from sklearn.neural_network._multilayer_perceptron import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#from dbn import DBN
#from rbm import RBM, shuffle_corpus, binary_data


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

    def _initialize(self, y, layer_units, dtype):
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
        self.out_activation_ = "logistic"

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(
                layer_units[i], layer_units[i + 1], dtype
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)
        
        # Do not modify ^^^^
        #-------------------------------------------------------


        # Modify/Overwrite coefs_ and intercepts_
        # initialization of layers with RBM/DBN layer weights
        print("Complete")
        if self.Ws is not None:
            self.coefs_=self.Ws
            print(self.coefs_)
            self.intercepts_=self.hbiases
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
