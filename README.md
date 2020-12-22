Date uploaded: Jan 11, 2020
Date updated: Dec 22, 2020
# fake-news-detection

## Intro
The goal of this repository is to build a machine learning model which is able to detect fake news by estimating the relative perspective (stance) of two pieces of text relative to a topic or claim.

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

Rationale: The related/unrelated classification task is expected to be much easier and is less relevant for detecting fake news, so it is given less weight in the evaluation metric. The Stance Detection task (classify as agrees, disagrees or discuss) is both more difficult and more relevant to fake news detection, so is to be given much more weight in the evaluation metric.

Concretely, if a [HEADLINE, BODY TEXT] pair in the test set has the target label unrelated, evaluation score will be incremented by 0.25 if it labels the pair as unrelated.
If the [HEADLINE, BODY TEXT] test pair is related, the evaluation score will be incremented by 0.25 if it labels the pair as any of the three classes: agrees, disagrees, or discusses.
The evaluation score will so be incremented by an additional 0.75 for each related pair if gets the relationship right by labeling the pair with the single correct class: agrees, disagrees, or discusses.

## Data

Training sets: Pairs of headline and body text with the appropriate class label for each.
- train_bodies.csv - shape (1683, 2): contains the article body column ('Body ID') with corresponding body text of article ('article Body') column.  
- train_stances.csv - shape (49952, 3): contains article headline ('Headline') for pairs of article body ('Body ID'), and labeled stance ('Stance') columns.

Test sets: Pairs of headline and body text without class labels used to evaluate systems.
- test_bodies.csv - shape (905, 2): contains the article body column ('Body ID') with corresponding body text of article ('article Body') column.  
- test_stances_unlabeled.csv - shape (25414, 3): contains article headline ('Headline') for pairs of article body ('Body ID') (no Stance column).

Data source: The data is derived from the Emergent Dataset created by Craig Silverman. For more information, visit [Fake news challenge](http://www.fakenewschallenge.org/).

## Models

I implemented 3 methods:

### 1. Perceptron
(44 features)

- Score on the development set (Hold-out split): 3307.25 out of 4448.5 (74.3%).
- Score on the test set: 8461.75 out of 11651.25 (72.6%).


### 2. Softmax linear model with additional tf-idf features
(44 features + tf-idf features)

- Score on the development set (Hold-out split): 3857.75 out of 4448.5 (86.7%).
- Score on the test set: 8963.25 out of 11651.25 (76.9%)


###  3. Feedforward neural network with additional tf-idf features
(44 features + tf-idf features)

- Score on the development set (Hold-out split): 3907.25 out of 4448.5 (**87.8%**).
- Scores on the test set: 8984.25 out of 11651.25	(**77.1%**).

## Usage

```
python main.py -method [method's name]
```
method's name: choose one method for training: perceptron, softmax_linear_model (softmax linear model with additional tf-idf features), or feedforward_NN (feedforward neural network with additional tf-idf features).

## Reference

Data source and evaluation: [Fake news challenge](http://www.fakenewschallenge.org/)
