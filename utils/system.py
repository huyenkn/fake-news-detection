import sys
import os
import re
import argparse

def parse_params():
    parser = argparse.ArgumentParser(description='FakeNewsChallenge fnc-1-baseline')
    parser.add_argument('-c', '--clean-cache', action='store_true', default=False, help="clean cache files")
    parser.add_argument('-method', type=str, choices=['perceptron', 'softmax_linear_model', 
        'feedforward_NN'], default='feedforward_NN',
        help='choose one method for training: perceptron, softmax_linear_model (softmax linear model with tf-idf features), or feedforward_NN (feedforward neural network with tf-idf features')

    params = parser.parse_args()

    if not params.clean_cache:
        return params

    dr = "features"
    for f in os.listdir(dr):
        if re.search('\.npy$', f):
            fname = os.path.join(dr, f)
            os.remove(fname)
    for f in ['hold_out_ids.txt', 'training_ids.txt']:
        fname = os.path.join('splits', f)
        if os.path.isfile(fname):
            os.remove(fname)
    print("All clear")

    return params

def check_version():
    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)
