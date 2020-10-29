"""
Classifies an embedding as it's closest neighbor in the support set.
If the distance to closest neighbor is more than a threshold it is classified as a new observation.
"""
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data import get_images, get_omniglot, get_embeddings
from config import IMG_SIZE, DESCRIPTOR_SIZE
from metrics import map_at_k, mean_accuracy_at_k, map_metric
import models
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.spatial import distance
from utils import show_accuracy_curve

# Datasets
FISH = 0
OMNIGLOT = 1

# Dataset flavours
FISH_HEAD = 0
FISH_WHOLE = 1
FISH_HEAD_LEFT = 2
FISH_HEAD_RIGHT = 3
FISH_PAIR_LEFT = 4
FISH_PAIR_RIGHT = 5
FISH_BINARY = 6

# Configs
BATCH_SIZE = 5
THRESHOLD = 0.55
DISTANCE_METRIC = 'euclidean'
GREYSCALE = False
RETRAIN = True
DEMO = False
USE_DATASET = FISH
DATASET_FLAVOUR = FISH_HEAD
IGNORE_NEW_OBSERVATIONS = True
EXPORT_PREDICTIONS = True
EXPORT_EMBEDDINGS = True

# Load model
model = models.triplet_network_ohnm
if DATASET_FLAVOUR == FISH_PAIR_LEFT:
    model.load_weights('./models/left')
elif DATASET_FLAVOUR == FISH_PAIR_RIGHT:
    model.load_weights('./models/right')
else:
    model.load_weights('./models/embedding')

# Load dataset
if USE_DATASET == FISH:

    direction_labels = True
    if DATASET_FLAVOUR in [FISH_PAIR_LEFT, FISH_PAIR_RIGHT]:
        direction_labels = False

    format = "jpg"
    if DATASET_FLAVOUR == FISH_BINARY:
        format = "png"

    if DATASET_FLAVOUR == FISH_HEAD:
        DATASET_DIR = os.path.join('..', 'data', 'dataset', 'cropped_head', 'direction')
        with open("../data/classes_direction.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_BINARY:
        DATASET_DIR = os.path.join('..', 'data', 'binary')
        with open("../data/classes_direction.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_HEAD_LEFT:
        DATASET_DIR = os.path.join('..', 'data', 'dataset', 'cropped_head', 'direction_left')
        with open("../data/classes_direction_left.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_PAIR_LEFT:
        DATASET_DIR = os.path.join('..', 'data', 'dataset', 'cropped_head', 'direction_left')
        with open("../data/classes_pairs.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_PAIR_RIGHT:
        DATASET_DIR = os.path.join('..', 'data', 'dataset', 'cropped_head', 'direction_right')
        with open("../data/classes_pairs.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_WHOLE:
        DATASET_DIR = os.path.join('..', 'data', 'dataset', 'cropped_body', 'direction')
        with open("../data/classes_direction.txt") as file:
            classes = [line.strip() for line in file]

    X_train, y_train, y_train_exact = get_images(os.path.join(DATASET_DIR, "train"), classes, IMG_SIZE,
                                                 greyscale=GREYSCALE,
                                                 direction_labels=direction_labels, format=format, exact_classes=True)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)

    X_test, y_test, y_test_exact = get_images(os.path.join(DATASET_DIR, "test"), classes, IMG_SIZE, greyscale=GREYSCALE,
                                              direction_labels=direction_labels, format=format, exact_classes=True)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)

if USE_DATASET == OMNIGLOT:
    X, y = get_omniglot()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Get filenames (used for exported predictions)
filenames = glob.glob(os.path.join(DATASET_DIR, "test", "*.jpg"))
filenames = [os.path.basename(filename) for filename in filenames]

drop_indices = []
# Set y to -1 if record does not exist in the training set
for i, y_test_i in enumerate(y_test):
    if y_test_i not in y_train:
        y_test[i] = -1
        drop_indices.append(i)

if IGNORE_NEW_OBSERVATIONS:
    # Remove new observations from dataset
    X_test = np.delete(X_test, drop_indices, axis=0)
    y_test = np.delete(y_test, drop_indices, axis=0)
    y_test_exact = np.delete(y_test_exact, drop_indices, axis=0)
    filenames = np.delete(filenames, drop_indices, axis=0)

if not IGNORE_NEW_OBSERVATIONS:
    # Need validation set for finding threshold
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

# Prints dataset stats
print("test set size: ", str(len(y_test)))
unique, counts = np.unique(y_test, return_counts=True)
new_observations = dict(zip(unique, counts))
print("test unique classes: ", len(unique))
new_observations = new_observations[-1] if -1 in new_observations else 0
print("new observations in test: ", str(new_observations))

# Stores support set embeddings into a dataframe
X_embeddings, y_embeddings = get_embeddings(model, X_train, y_train)
columns = ["x" + str(i) for i in range(DESCRIPTOR_SIZE)] + ["y", "support_exact"]
df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings, y_train_exact]), columns=columns)
df[columns[:-2]] = df[columns[:-2]].apply(pd.to_numeric)
df = df.astype({"y": int})

if EXPORT_EMBEDDINGS:
    X_embeddings_test, y_embeddings_test = get_embeddings(model, X_test, y_test)
    columns = ["x" + str(i) for i in range(DESCRIPTOR_SIZE)] + ["y"]
    df2 = pd.DataFrame(np.column_stack([X_embeddings_test, y_embeddings_test]), columns=columns)
    df2 = df2.astype({"y": int})
    df2["set"] = "test"
    df3 = df.copy()
    df3["set"] = "train"
    df2 = pd.concat([df2, df3])
    df2.to_csv("embeddings.csv", index=False)


def predict_k(df_embeddings, x_test, k=1, metric='euclidean', predict_new=True, threshold=0.5, exact_class=False,
              y_test=None):
    """Predicts the classes for x_test from the embeddings dataframe."""

    embeddings = model.predict(x_test)

    support_exact = []
    support_exact_at_correct = []
    y_pred_exact = []
    correct_exact = []
    y_pred = []

    for i, embedding in enumerate(embeddings):
        if metric == 'euclidean':
            df_embeddings['distance'] = np.linalg.norm(df_embeddings.iloc[:, :DESCRIPTOR_SIZE].sub(embedding), axis=1)
        elif metric == 'cosine':
            df_embeddings['distance'] = df_embeddings.apply(
                lambda row: distance.cosine(embedding, row[:DESCRIPTOR_SIZE]), axis=1)
        df_sorted = df_embeddings.copy()
        if predict_new:
            df_sorted = df_sorted.append({'y': -1, 'distance': threshold}, ignore_index=True)
        df_sorted = df_sorted.sort_values(by='distance', ignore_index=True)  # TODO: Indent?
        k_best = df_sorted['y'][:k].tolist()
        y_pred.append(k_best)

        if exact_class:
            y_pred_exact.append(y_test_exact[i])
            support_exact.append(df_sorted["support_exact"][0])
            if y_test is not None:
                first_correct = df_sorted[df_sorted['y'] == y_test[i]]
                correct_exact.append(first_correct["support_exact"].to_numpy()[0])
                support_exact_at_correct.append(df_sorted[df_sorted["y"] == y_test[i]]["support_exact"].to_numpy()[0])

    if exact_class:
        print(len(support_exact_at_correct))
        return np.array(y_pred), np.array(y_pred_exact), np.array(support_exact), np.array(correct_exact), np.array(support_exact_at_correct)
    else:
        return np.array(y_pred)


def rank(y_pred, y_true):
    """Gets the retrieval ranks of y_pred."""
    r = []
    for xi, yi in zip(y_pred, y_true):
        r.append(list(xi).index(yi) + 1)
    return r


def evaluate(y_pred, y_true, accuracy_curve=False):
    """Evaluates y_pred given y_true."""
    new_count = list(y_pred[:, 0]).count(-1)
    accuracy = accuracy_score(y_true, y_pred[:, 0])
    f1 = f1_score(y_true, y_pred[:, 0], average='macro')
    mAP_5 = map_at_k(y_test, y_pred, 5)
    mAcc_5 = mean_accuracy_at_k(y_test, y_pred, 5)
    print("Accuracy: ", accuracy, "\nmAcc@5: ", mAcc_5, "\nMAP@5: ", mAP_5, "\nF1: ", f1, "\nNew: ", new_count)

    if accuracy_curve:
        auc = show_accuracy_curve(y_pred, y_test, return_auc=True)
        print("AUC: ", auc)


# Find best threshold
if not IGNORE_NEW_OBSERVATIONS:
    best_f1 = 0
    best_t = -1
    for t in [0.7, 0.75, 0.8, 0.85, 0.9]:
        print("t = ", t)
        y_pred = predict_k(df, X_test, metric=DISTANCE_METRIC, predict_new=True, threshold=t)
        new_count = list(y_pred[:, 0]).count(-1)
        accuracy = accuracy_score(y_test, y_pred[:, 0])
        f1 = f1_score(y_test, y_pred[:, 0], average='macro')
        print("Accuracy: ", accuracy, "\tF1: ", f1, "\tNew: ", new_count)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
        print("======================================")
    print("best t: ", best_t)
else:
    best_t = THRESHOLD

y_pred, y_pred_exact, support_exact, correct_exact, support_exact_at_correct = predict_k(df, X_test, k=len(df), metric=DISTANCE_METRIC,
                                                               predict_new=not IGNORE_NEW_OBSERVATIONS,
                                                               threshold=best_t, exact_class=True, y_test=y_test)

if EXPORT_PREDICTIONS:
    ranks = rank(y_pred, y_test)
    columns = ["filename"] + ["p" + str(i + 1) for i in range(len(df))] + ["support_exact", "query_exact",
                                                                           "correct_exact", "support_exact_at_correct", "rank", "y"]
    df_pred = pd.DataFrame(
        np.column_stack((filenames, y_pred, support_exact, y_pred_exact, correct_exact, support_exact_at_correct, ranks, y_test)),
        columns=columns)
    df_pred.to_csv("predictions.csv", index=False)

evaluate(y_pred, y_test, accuracy_curve=True)

# Examples of predictions
fig, ax = plt.subplots(nrows=4, ncols=2)
for i, row in enumerate(ax):
    row[0].imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    pred_class = -1 if y_pred[i][0] == -1 else classes[int(y_pred[i][0])]
    true_class = -1 if y_pred[i][0] == -1 else classes[y_test[i]]
    row[1].text(0, 0, "Predicted: " + str(pred_class))
    row[1].text(0, 0.5, "Ground truth: " + str(true_class))  # Check if y is -1
plt.show()
