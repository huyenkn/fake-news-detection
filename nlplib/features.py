from nlplib.constants import OFFSET
import numpy as np
import torch
from heapq import nlargest

def get_top_features_for_label_numpy(weights,label,k=5):
    '''
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''

    #raise NotImplementedError
    feature_dict = {}

    for i in weights:
        if label == i[0]:
            feature_dict[i] = weights[i]

    features = list(feature_dict.items())
    top_features = sorted(features, key=lambda x: -x[1])[:k]
    return top_features

def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''
    top_list = []
    label_dict = dict(zip(label_set, range(4)))
    label = label_dict[label]

    weights = model.Linear.weight
    vocab = sorted(vocab)
    values, indices = weights.topk(k, dim=1)
    for i in range(len(label_set)):
        if i == label:
            for j in indices[i]:
                top_list.append(vocab[j])
    return top_list

    #raise NotImplementedError

def get_token_type_ratio(counts):
    '''
    compute the ratio of tokens to types

    :param counts: bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types
    :rtype: float

    '''
    tokens_type_ratio = counts.sum()/(counts!=0).sum()
    return tokens_type_ratio
    
    #raise NotImplementedError

def concat_ttr_binned_features(data):
    '''
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    '''
    bins = np.zeros((data.shape[0], 7))
    for i in range(data.shape[0]):
        if (data[i]!=0).sum() != 0:
            if get_token_type_ratio(data[i]) < 6:
                bins[i, int(get_token_type_ratio(data[i]))] = 1
            else:
                bins[i, 6] = 1
    data = np.hstack((data, bins))
    return data

    #raise NotImplementedError
