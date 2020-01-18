from nlplib.constants import OFFSET
from nlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict
from collections import Counter
import numpy as np

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counters = Counter()
    for i in range(len(y)):
        if y[i] == label:
            corpus_counters += x[i]
    return defaultdict(float, corpus_counters)

    #raise NotImplementedError

def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''

    # raise NotImplementedError
    corpus_dict = get_corpus_counts(x,y,label)
    vocab_dict = defaultdict(float)
    for word in vocab:
        vocab_dict[word] = corpus_dict[word] + smoothing
        vocab_dict[word] = np.log(vocab_dict[word]/(sum(corpus_dict.values()) + smoothing*len(vocab)))
    return vocab_dict

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    
    labels = set(y)
    weights = defaultdict(float)
    counts = Counter()

    for cnter in x:
        counts += cnter
    vocab = counts.keys()
    log_py = defaultdict(float)

    for label in labels:
        log_py[label] = np.log(sum(y == label)/len(y))
   
    for label in labels:
        log_pxy = estimate_pxy(x,y,label,smoothing,vocab)
        for word in log_pxy:
            weights[(label, word)] = log_pxy[word]
        weights[(label, '**OFFSET**')] = log_py[label]
    return weights

    #raise NotImplementedError

def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    #raise NotImplementedError
    scores = {}
    max_score = -1e9
    best_smoother = None
    best_theta_nb = None
    for smoother in smoothers:
        theta_nb = estimate_nb(x_tr,y_tr,smoother)
        y_hat = clf_base.predict_all(x_dv,theta_nb,y_dv)
        score = evaluation.acc(y_hat,y_dv)
        if score > max_score:
            max_score = score
            best_smoother = smoother
            best_theta_nb = theta_nb

    return best_smoother, best_theta_nb

    """
    max_score = max(scores.values())
    
    for k, v in scores.items():
        if v == max_score:
            best_smoother = k
    return best_smoother, scores
    """
    



