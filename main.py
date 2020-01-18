import sys
import argparse
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from feature_engineering import clean
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
from collections import Counter

from nlplib import perceptron
from nlplib import clf_base
from nlplib import logreg

import torch
from torch import optim

from tf_idf import compute_tfidf_vector, create_vocab, tfidf_matrix


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    #all arrays
    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]

    return X,y

def dense_to_sparse(X):
        
    X_list = []
    for i in range(X.shape[0]):
        X_Cnt = Counter()
        for j in range(X.shape[1]):
            X_Cnt[j] = X[i, j]
        X_list.append(X_Cnt)
    return X_list

def perceptron_mt():

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_train_sparse = dense_to_sparse(X_train)
        y_train_sparse = y_train.tolist()
        theta_perc,theta_perc_history = perceptron.estimate_perceptron(X_train_sparse,y_train_sparse,10)

        X_test = Xs[fold]
        y_test = ys[fold]

        X_test_sparse = dense_to_sparse(X_test)
        labels = set(y_train_sparse)

        y_predicted_num = clf_base.predict_all(X_test_sparse,theta_perc,labels)
        print('Evaluating', '...')

        y_predicted = [LABELS[int(a)] for a in y_predicted_num]
        y_actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(y_actual, y_predicted)
        max_fold_score, _ = score_submission(y_actual, y_actual)

        best_score = 0
        best_theta = None

        score = fold_score/max_fold_score
        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_theta = theta_perc


    #Run on Holdout set and report the final score on the holdout set
    X_holdout_sparse = dense_to_sparse(X_holdout)
    y_predicted_num = clf_base.predict_all(X_holdout_sparse,best_theta,labels)
    y_actual_holdout = [LABELS[int(a)] for a in y_holdout]
    y_predicted_holdout = [LABELS[int(a)] for a in y_predicted_num]

    #Run on competition dataset
    X_competition_sparse = dense_to_sparse(X_competition)
    y_predicted_num = clf_base.predict_all(X_competition_sparse,best_theta,labels)
    y_actual_compe = [LABELS[int(a)] for a in y_competition]
    y_predicted_compe = [LABELS[int(a)] for a in y_predicted_num]

    return y_actual_holdout, y_predicted_holdout, y_actual_compe, y_predicted_compe

def max_likelihood_est_with_tfidf_features(train_dataset):

    bodies = train_dataset.articles

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        head_corpus = []
        body_corpus = []
        for i in ids:
            for d in fold_stances[i]:
                head_corpus.append(clean(d['Headline']))
                body_corpus.append(clean(bodies[d['Body ID']]))
        print('---------')

        vocab = create_vocab(head_corpus, body_corpus, min_freq=5)
        print('Vocab size:', len(vocab))


        # # create tfidf feature vector
        train_tfidf_head = compute_tfidf_vector(head_corpus, vocab)
        train_tfidf_body = compute_tfidf_vector(body_corpus, vocab)
        
        # concat x_train with tfidf feature vector 
        #print(X_train.shape)
        #print(train_tfidf_head.shape)
        #print(train_tfidf_body.shape)
        X_train = np.c_[X_train, train_tfidf_head, train_tfidf_body]
        print(X_train.shape)
        print('+++++++++++++++')

        test_tfidf_head, test_tfidf_body = tfidf_matrix(fold_stances[fold], bodies, vocab)

        X_test = Xs[fold]
        y_test = ys[fold]

        y_test_np = np.array(y_test)
        X_test_np = np.c_[X_test, test_tfidf_head, test_tfidf_body]

        torch.manual_seed(765)
        model = logreg.build_linear(X_train, y_train, args.method)
        model.add_module('softmax', torch.nn.LogSoftmax(dim=1))

        loss = torch.nn.NLLLoss()

        print('---------------')

        model_trained, losses, accuracies = logreg.train_model_batch(
                                                       loss, 
                                                       model,
                                                       X_train,
                                                       y_train,
                                                       X_dv=X_test_np,
                                                       Y_dv=y_test_np,
                                                       num_its=1,
                                                       optim_args={'lr':1e-3})

        X_test_var = torch.from_numpy(X_test_np).float()
        _, y_predicted_var = model_trained.forward(X_test_var).max(dim=1)
        print(type(y_predicted_var))

        print('Evaluating', '...')

        y_predicted = [LABELS[int(a)] for a in y_predicted_var]
        y_actual = [LABELS[int(a)] for a in y_test_np]

        fold_score, _ = score_submission(y_actual, y_predicted)
        max_fold_score, _ = score_submission(y_actual, y_actual)

        best_score = 0
        best_model = None

        score = fold_score/max_fold_score
        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_model = model_trained
            best_vocab = vocab


    #Run on Holdout set and report the final score on the holdout set
    holdout_tfidf_head, holdout_tfidf_body = tfidf_matrix(hold_out_stances, bodies, best_vocab)

    X_holdout_combi = np.c_[X_holdout, holdout_tfidf_head, holdout_tfidf_body]

    X_holdout_var = torch.from_numpy(X_holdout_combi).float()
    y_holdout_np = np.array(y_holdout)
    y_holdout_var = torch.from_numpy(y_holdout_np)

    _, y_predicted_var = best_model.forward(X_holdout_var).max(dim=1)
    y_actual_holdout = [LABELS[int(a)] for a in y_holdout_var]
    y_predicted_holdout = [LABELS[int(a)] for a in y_predicted_var]

    #Run on competition DataSet
    compe_bodies = competition_dataset.articles
    compe_tfidf_head, compe_tfidf_body = tfidf_matrix(competition_dataset.stances, compe_bodies, best_vocab)

    X_competition_combi = np.c_[X_competition, compe_tfidf_head, compe_tfidf_body]

    X_competition_var = torch.from_numpy(X_competition_combi).float()
    y_competition_np = np.array(y_competition)
    y_competition_var = torch.from_numpy(y_competition_np)
    _, y_predicted_var = best_model.forward(X_competition_var).max(dim=1)
    y_actual_compe = [LABELS[int(a)] for a in y_competition_var]
    y_predicted_compe = [LABELS[int(a)] for a in y_predicted_var]

    return y_actual_holdout, y_predicted_holdout, y_actual_compe, y_predicted_compe


if __name__ == "__main__":
    check_version()
    args = parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    #folds: list(10), list(len=134, 135) each; hold_out: list(337)
    folds,hold_out = kfold_split(d,n_folds=10)


    #fold_stances (defaultdict: 10 keys, hold_out_stances: list (9622) of ordereddict
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, 
        competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    #X_holdout: ndarray, (9622, 44-features); y_holdout: list (9622- 0->3: labels)
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")

    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    if args.method == 'perceptron':
        print('Running ' + args.method + '...')
        y_actual_holdout, y_predicted_holdout, y_actual_compe, y_predicted_compe = perceptron_mt()

    else:
        y_actual_holdout, y_predicted_holdout, y_actual_compe, y_predicted_compe = max_likelihood_est_with_tfidf_features(d)

    print("Scores on the dev set")
    report_score(y_actual_holdout, y_predicted_holdout)
    print("")
    print("")

    print("Scores on the test set")
    report_score(y_actual_compe, y_predicted_compe)

