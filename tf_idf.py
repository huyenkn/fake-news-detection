from utils.dataset import DataSet
from feature_engineering import clean
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 

from collections import Counter

def compute_tfidf_vector(corpus, vocab):

	#use our own vocabulary set
	vectorizer = TfidfVectorizer(vocabulary=vocab)
	X = vectorizer.fit_transform(corpus)
	X_dense = X.todense()
	return X_dense

def counter_from_text_list(text_list):

	word_list = []
	for text in text_list:
		text_words = text.split(' ')
		word_list += text_words
	#return word_list
	corpus = []
	word_cnter = Counter()
	for w in word_list:
		word_cnter[w] += 1

	return word_cnter


def create_vocab(headline, body, min_freq=5):

	# for d in headlines:
	#   body_corpus.append(clean(bodies[d['Body ID']]))
	#   headline_corpus.append(clean(d['Headline']))
	#   body = body_corpus[-1].split(' ')
	#   head = headline_corpus[-1].split(' ')

	#vocab.update(body)
	#vocab.update(head)
	#head_words = split_words(headline)
	#body_words = split_words(body)

	#for w in body_words:
	#   body_cnt[w] += 1

	#for w in head_words:
	#   headline_cnt[w] += 1

	body_cnt = counter_from_text_list(body)
	headline_cnt = counter_from_text_list(headline)

	combi_cnt = body_cnt + headline_cnt
	combi_pruned = Counter({k:v for k,v in combi_cnt.items() if v >= min_freq})
	vocab = combi_pruned.keys()
	return vocab


def tfidf_matrix(list_of_dicts, bodies, vocab):
	# input: list of Ordereddict (keys includes: Headline, Body ID, and Stance)
	head_corpus = []
	body_corpus = []

	for d in list_of_dicts:
		head_corpus.append(clean(d['Headline']))
		body_corpus.append(clean(bodies[d['Body ID']]))
	#print('---------')

	# vocab = create_vocab(head_corpus, body_corpus, min_freq=5)
	#print('Vocab size:', len(vocab))

	tfidf_headline = compute_tfidf_vector(head_corpus, vocab)
	tfidf_body = compute_tfidf_vector(body_corpus, vocab)
	
	#create tfidf feature vector
	return tfidf_headline, tfidf_body

