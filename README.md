# fake-news-detection-nlp

## Intro
This repository is to detect fake news by estimating the relative perspective (stance) of two pieces of text relative to a topic or claim.

Input:
A headline and a body text - either from the same news article or from two different articles.

Output:
Classify the stance of the body text relative to the claim made in the headline into one of four categories:
Agrees: The body text agrees with the headline.
Disagrees: The body text disagrees with the headline.
Discusses: The body text discuss the same topic as the headline, but does not take a position
Unrelated: The body text discusses a different topic than the headline

## Evaluation
Performance is measured based on a weighted, two-level scoring system:

Level 1: Classify headline and body text as related or unrelated: 25% score weighting
Level 2: Classify related pairs as agrees, disagrees, or discusses: 75% score weighting

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

Scores on the development set (Hold-out split for development)
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |     0     |     3     |    704    |    55     |
-------------------------------------------------------------
| disagree  |     0     |     0     |    158    |     4     |
-------------------------------------------------------------
|  discuss  |     0     |    13     |   1674    |    113    |
-------------------------------------------------------------
| unrelated |     0     |     2     |   1241    |   5655    |
-------------------------------------------------------------
Score: 3307.25 out of 4448.5	(74.34528492750366%)

Scores on the test set
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |     0     |    10     |   1726    |    167    |
-------------------------------------------------------------
| disagree  |     0     |    10     |    551    |    136    |
-------------------------------------------------------------
|  discuss  |     1     |    25     |   4014    |    424    |
-------------------------------------------------------------
| unrelated |     0     |    26     |   2885    |   15438   |
-------------------------------------------------------------
Score: 8461.75 out of 11651.25	(72.62525480098702%)


### 2. Softmax linear model with additional tf-idf features
(44 features + tf-idf features)

Scores on the development set (Hold-out split for development)
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    442    |     6     |    262    |    52     |
-------------------------------------------------------------
| disagree  |    61     |    29     |    62     |    10     |
-------------------------------------------------------------
|  discuss  |    114    |     2     |   1557    |    127    |
-------------------------------------------------------------
| unrelated |    13     |     1     |    72     |   6812    |
-------------------------------------------------------------
Score: 3857.75 out of 4448.5	(86.72024277846465%)

Scores on the test set
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    735    |     0     |    846    |    322    |
-------------------------------------------------------------
| disagree  |    156    |     2     |    285    |    254    |
-------------------------------------------------------------
|  discuss  |    446    |     0     |   3312    |    706    |
-------------------------------------------------------------
| unrelated |    54     |     1     |    370    |   17924   |
-------------------------------------------------------------
Score: 8963.25 out of 11651.25	(76.9295140006437%)


###  3. Feedforward neural network with additional tf-idf features
(44 features + tf-idf features)

Scores on the development set (Hold-out split for development)
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    445    |    10     |    259    |    48     |
-------------------------------------------------------------
| disagree  |    28     |    78     |    49     |     7     |
-------------------------------------------------------------
|  discuss  |    93     |    30     |   1565    |    112    |
-------------------------------------------------------------
| unrelated |    13     |     1     |    76     |   6808    |
-------------------------------------------------------------
Score: 3907.25 out of 4448.5	(87.8329774081151%)

Scores on the test set
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    599    |     3     |    987    |    314    |
-------------------------------------------------------------
| disagree  |    127    |     9     |    302    |    259    |
-------------------------------------------------------------
|  discuss  |    320    |     4     |   3476    |    664    |
-------------------------------------------------------------
| unrelated |    59     |     1     |    431    |   17858   |
-------------------------------------------------------------
Score: 8984.25 out of 11651.25	(77.10975217251368%)

## Usage

```
python main.py -method [method name]
```
method names: choose one method for training: perceptron, softmax_linear_model (softmax linear model with additional tf-idf features), or feedforward_NN (feedforward neural network with additional tf-idf features).





















