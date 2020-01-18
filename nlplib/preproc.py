from collections import Counter

import pandas as pd
import numpy as np

def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    #raise NotImplementedError
    text_list = text.split()
    return Counter(text_list)

def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counts = Counter()
    for bag in bags_of_words:
        counts += bag
    return counts

def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    set1 = set(bow1.keys())
    set2 = set(bow2.keys())
    return set1 - set2

    #raise NotImplementedError

def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    vocab_dict = { k : training_counts[k] for k in training_counts 
        if training_counts[k] >= min_counts}
    vocab = set(vocab_dict)
    new_target_data = []
    for cnt in target_data:
        new_cnt = Counter()
        for word in cnt:
            if word in vocab:
                new_cnt[word] = cnt[word]
        new_target_data.append(new_cnt)
    return new_target_data, vocab   
   
    # vocab_dict = { k : training_counts[k] for k in training_counts 
    #     if training_counts[k] >= min_counts}

    # vocab = set(vocab_dict)

    # for cnt in target_data:
    #     new_cnt = list(cnt)
    #     for word in new_cnt:
    #         if word not in vocab:
    #             del cnt[word]
    #raise NotImplementedError

def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''

    word_matrix = np.zeros((len(bags_of_words), len(vocab)))
    vocab = sorted(vocab)
    for i in range(len(bags_of_words)):
        for j in range(len(vocab)):
            word_matrix[i, j] = bags_of_words[i][vocab[j]]
    
    return word_matrix

    #raise NotImplementedError

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())
