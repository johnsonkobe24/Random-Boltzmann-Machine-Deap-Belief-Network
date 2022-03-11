import os
import pickle
import numpy as np
from numpy.testing import assert_allclose

import unittest
import gradescope_utils.autograder_utils.json_test_runner as json
from gradescope_utils.autograder_utils.decorators import weight, visibility

from rbm import RBM
#from solution_rbm import RBM

SEED = 10417617
TOLERANCE = 1e-5

# to run one test: python -m unittest tests.h_v
# to run all tests: python -m unittest tests

with open('./tests.pk',"rb") as f:
    tests = pickle.load(f)


class h_v(unittest.TestCase):
    @weight(2)
    def test(self):
        # Instantiate
        np.random.seed(SEED)
        rbm = RBM(n_visible=784, n_hidden=4, k=3, lr=0.01, max_epochs=None)
        v = np.random.randint(2, size=(784,))

        # Test
        h_prob = rbm.h_v(v)
        assert_allclose(h_prob, tests["h_v"], atol=TOLERANCE)
        
        
class sample_h(unittest.TestCase):
    @weight(2)
    def test(self):    
        # Instantiate
        np.random.seed(SEED)
        rbm = RBM(n_visible=784, n_hidden=4, k=3, lr=0.01, max_epochs=None)
        h_prob = np.random.uniform(low=0.0, high=1.0, size=(4,))

        # Test
        h_sample = rbm.sample_h(h_prob)
        assert_allclose(h_sample, tests["sample_h"], atol=TOLERANCE)
        
        
class v_h(unittest.TestCase):
    @weight(2)
    def test(self):
        # Instantiate
        np.random.seed(SEED)
        rbm = RBM(n_visible=784, n_hidden=4, k=3, lr=0.01, max_epochs=None)
        h = np.random.randint(2, size=(4,))
        
        # Test
        v_prob = rbm.v_h(h)
        assert_allclose(v_prob, tests["v_h"], atol=TOLERANCE)  
        
class sample_v(unittest.TestCase):
    @weight(2)
    def test(self):    
        # Instantiate
        np.random.seed(SEED)
        rbm = RBM(n_visible=784, n_hidden=4, k=3, lr=0.01, max_epochs=None)
        v_prob = np.random.uniform(low=0.0, high=1.0, size=(784,))

        # Test
        v_sample = rbm.sample_v(v_prob)
        assert_allclose(v_sample, tests["sample_v"], atol=TOLERANCE)     
        
        
class sample_v_injection(unittest.TestCase):
    @weight(5)
    def test(self):        
        # Instantiate
        np.random.seed(SEED)
        rbm = RBM(n_visible=784, n_hidden=4, k=3, lr=0.01, max_epochs=None)

        v_prob = np.random.uniform(low=0.0, high=1.0, size=(784,))
        v_true = np.random.randint(2, size=(784,))
        v_observation = np.zeros((784,))
        v_observation[:392] = 1

        # Test
        v_sample_injection = rbm.sample_v(v_prob=v_prob, 
                                          v_true=v_true,
                                          v_observation=v_observation)
        assert_allclose(v_sample_injection, tests["v_sample_injection"], atol=TOLERANCE)

class gibbs_k(unittest.TestCase):
    @weight(2)
    def test(self):
        # Instantiate
        np.random.seed(SEED)
        rbm = RBM(n_visible=784, n_hidden=4, k=3, lr=0.01, max_epochs=None)
        v = np.random.randint(2, size=(784,))
        h0, v0, h_sample, v_sample, h_prob, v_prob = rbm.gibbs_k(v)
        
        # Test
        print(' h_sample.shape', h_sample.shape)
        print(' v_sample.shape', v_sample.shape)
        print(' h_prob.shape', h_prob.shape)
        print(' v_prob.shape', v_prob.shape)
        assert_allclose(h_sample, tests["gibbs_k"]["h_sample"], atol=TOLERANCE)
        assert_allclose(h_prob,   tests["gibbs_k"]["h_prob"],   atol=TOLERANCE)
        assert_allclose(v_sample, tests["gibbs_k"]["v_sample"], atol=TOLERANCE)
        assert_allclose(v_prob,   tests["gibbs_k"]["v_prob"],   atol=TOLERANCE)

class rec_error(unittest.TestCase):
    @weight(2)
    def test(self):        
        # Instantiate
        np.random.seed(SEED)
        rbm = RBM(n_visible=784, n_hidden=4, k=3, lr=0.01, max_epochs=None)
        V = np.random.randint(2, size=(5,784))
        
        # Test
        rec_error = rbm.evaluate(V)
        assert_allclose(rec_error, tests["rec_error"], atol=TOLERANCE)
        
class update(unittest.TestCase):
    @weight(3)
    def test(self):        
        # Instantiate
        np.random.seed(SEED)
        rbm = RBM(n_visible=784, n_hidden=4, k=3, lr=0.01, max_epochs=None)
        v = np.random.randint(2, size=(784, ))
        
        # Test
        rbm.update(v)
        
        print(' rbm.W.shape', rbm.W.shape)
        print(' rbm.hbias.shape', rbm.hbias.shape)
        print(' rbm.vbias.shape', rbm.vbias.shape)
        assert_allclose(rbm.W,     tests["update"]["W"], atol=TOLERANCE)
        assert_allclose(rbm.hbias, tests["update"]["hbias"], atol=TOLERANCE)
        assert_allclose(rbm.vbias, tests["update"]["vbias"], atol=TOLERANCE)

