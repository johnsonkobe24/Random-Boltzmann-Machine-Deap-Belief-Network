import unittest
import numpy
import gradescope_utils.autograder_utils.json_test_runner as json
import os

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests')
    json.JSONTestRunner(visibility='visible').run(suite)
