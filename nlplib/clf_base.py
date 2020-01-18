from nlplib.constants import OFFSET
import numpy as np

def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# This will no longer work since python3's max does not guarantee deterministic ordering
# argmax = lambda x : max(x.items(),key=lambda y : y[1])[0]

def make_feature_vector(base_features,label):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    '''
    feature_vector = {}
    for i in set(base_features):
        feature_vector[(label, i)] = base_features[i]
    feature_vector[(label, '**OFFSET**')] = 1
    return feature_vector

    #use set(base_features) faster than base_features.keys()

def predict(base_features,weights,labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    '''
    scores = {}
    for label in labels:
        fv = make_feature_vector(base_features,label) 
        """"
        a = fv
        b = weights
        if len(b) < len(a):
            tmp = a
            a = b
            b = tmp
        for (x, y) in a:
            if x == label:
                scores[x] += a[x, y] * b.get((x, y), 0)
        """
        scores[label] = 0 
        if len(fv) > len(weights):
            for (x, y) in weights:
                if x == label:   
                    scores[x] += weights[x, y] * fv.get((x, y), 0)
        else:
            for (x, y) in fv:
                if x == label:
                    scores[x] += fv[x, y] * weights.get((x, y), 0)       


    """
    from collections import defaultdict
    scores = defaultdict(float)
    remove: scores[label] = 0
    
    """
    return argmax(scores),scores

def predict2(base_features,weights,labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    '''
    scores = {}
    for label in labels:
        fv = make_feature_vector(base_features,label) 

        """"
        a = fv
        b = weights
        if len(b) < len(a):
            tmp = a
            a = b
            b = tmp
        for (x, y) in a:
            if x == label:
                scores[x] += a[x, y] * b.get((x, y), 0)
        """
        scores[label] = 0 
        if len(fv) > len(weights):
            for (x, y) in weights:
                if x == label:   
                    scores[x] += weights[x, y] * fv.get((x, y), 0)
        else:
            for (x, y) in fv:
                if x == label:
                    scores[x] += fv[x, y] * weights.get((x, y), 0)       


    """
    from collections import defaultdict
    scores = defaultdict(float)
    remove: scores[label] = 0
    
    """
    return argmax(scores),scores


def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    #y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])

    y_hat = []
    for x_i in x:
        prediction, scores = predict2(x_i, weights, labels)
        y_hat.append(prediction)
        #print(scores)
    return y_hat