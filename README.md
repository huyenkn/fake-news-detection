# fake-news-detection-nlp

## Intro
This repository is to detect fake news by estimating the relative perspective (stance) of two pieces of text relative to a topic or claim.

Input:
A headline and a body text - either from the same news article or from two different articles.

Output:
Classify the stance of the body text relative to the claim made in the headline into one of four categories:
- Agrees: The body text agrees with the headline.
- Disagrees: The body text disagrees with the headline.
- Discusses: The body text discuss the same topic as the headline, but does not take a position.
- Unrelated: The body text discusses a different topic than the headline.

## Evaluation
Performance is measured based on a weighted, two-level scoring system:

- Level 1: Classify headline and body text as related or unrelated: 25% score weighting.
- Level 2: Classify related pairs as agrees, disagrees, or discusses: 75% score weighting.

(The related/unrelated classification task is expected to be easier and is less relevant for detecting fake news, so it is given less weight in the evaluation metric.)

## Data

Training sets: 
- train_bodies.csv - shape (1683, 2): contains the article body column ('Body ID') with corresponding body text of article ('article Body') column.  
- train_stances.csv - shape (49952, 3): contains article headline ('Headline') for pairs of article body ('Body ID'), and labeled stance ('Stance') columns.

Test sets:
- test_bodies.csv - shape (905, 2): contains the article body column ('Body ID') with corresponding body text of article ('article Body') column.  
- test_stances_unlabeled.csv - shape (25414, 3): contains article headline ('Headline') for pairs of article body ('Body ID') (no Stance column).

Data source: [Fake news challenge](http://www.fakenewschallenge.org/)

## Hold-out set split
Data are splitted using the generate_hold_out_split() function. This function ensures that the article bodies in the training set (fold set) are not present in the hold-out set (development set). 

## k-fold split
The training set (fold set) is also splitted into k folds (k=10) using the kfold_split function. 

## Models

Implemented 3 methods:

### 1. Perceptron
(44 features)

- Score on the development set (Hold-out split): 3307.25 out of 4448.5 (74.34528492750366%).
- Score on the test set: 8461.75 out of 11651.25 (72.62525480098702%).


### 2. Softmax linear model with additional tf-idf features
(44 features + tf-idf features)

- Score on the development set (Hold-out split): 3857.75 out of 4448.5 (86.72024.277846465%).
- Score on the test set: 8963.25 out of 11651.25 (76.9295140006437%)


###  3. Feedforward neural network with additional tf-idf features
(44 features + tf-idf features)

- Score on the development set (Hold-out split): 3907.25 out of 4448.5 (87.8329774081151%).
- Scores on the test set: 8984.25 out of 11651.25	(77.10975217251368%).

## Usage

```
python main.py -method [method's name]
```
method's name: choose one method for training: perceptron, softmax_linear_model (softmax linear model with additional tf-idf features), or feedforward_NN (feedforward neural network with additional tf-idf features).





















