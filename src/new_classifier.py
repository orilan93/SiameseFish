"""
Predicts whether a query sample exists in the support set or is a new unobserved sample.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import const
from data import get_omniglot, get_embeddings, load_dataset
from config import DESCRIPTOR_SIZE
import models
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Configs
THRESHOLD = 0.4
DEMO = False
USE_DATASET = const.FISH
DATASET_FLAVOUR = const.FISH_HEAD
SCORE_FUNCTION = 'F1'  # F1, TPR

model = models.triplet_network_ohnm
model.load_weights('../models/embedding')

if USE_DATASET == const.FISH:
    X_train, y_train, X_test, y_test, classes = load_dataset(DATASET_FLAVOUR, preprocess=True)

if USE_DATASET == const.OMNIGLOT:
    X, y = get_omniglot()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

drop_indices = []
# Set y to -1 if record does not exist in the training set
for i, y_test_i in enumerate(y_test):
    if y_test_i not in y_train:
        y_test[i] = -1
    else:
        y_test[i] = 1

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

print("test set size: ", str(len(y_test)))
unique, counts = np.unique(y_test, return_counts=True)
new_observations = dict(zip(unique, counts))
print("test unique classes: ", len(unique))
new_observations = new_observations[-1] if -1 in new_observations else 0
print("y_test new observations: ", str(new_observations))

X_embeddings, y_embeddings = get_embeddings(model, X_train, y_train)
columns = ["x" + str(i) for i in range(DESCRIPTOR_SIZE)] + ["y"]
df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
df = df.astype({"y": int})


def predict(x, y_true):
    embeddings = model.predict(x)
    y_list = []
    for i, embedding in enumerate(embeddings):
        df['distance'] = np.linalg.norm(df.iloc[:, :DESCRIPTOR_SIZE].sub(embedding), axis=1)
        idx = df['distance'].idxmin()
        distance = df['distance'][idx]
        if distance < THRESHOLD:
            y_pred = 1
        else:
            y_pred = -1
        y_list.append(y_pred)
    return y_list


def evaluate(x, y, report=False):
    prediction = predict(x, y)
    new_count = prediction.count(-1)
    accuracy = accuracy_score(y, prediction)
    if SCORE_FUNCTION == 'F1':
        score = f1_score(y, prediction, average='macro')
    if SCORE_FUNCTION == 'TPR':
        true_count = dict(zip(*np.unique(y, return_counts=True)))[-1]
        gt = y.copy()
        gt[gt == 1] = 0
        true_positive = (np.array(prediction) == gt).sum()
        score = true_positive / true_count
    print("Accuracy: ", accuracy, "\t", SCORE_FUNCTION, ": ", score, "\t New: ", new_count)
    if report:
        print(classification_report(y, prediction))
    return accuracy, score


best_score = 0
best_t = -1
for t in np.arange(0.7, 0.85, 0.01):
    print("t = ", t)
    THRESHOLD = t
    accuracy, score = evaluate(X_valid, y_valid)
    if score > best_score:
        best_score = score
        best_t = t
    print("======================================")

print("best t: ", best_t)

THRESHOLD = best_t
accuracy, score = evaluate(X_test, y_test)
