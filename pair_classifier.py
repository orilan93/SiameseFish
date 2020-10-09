"""
A classifier that uses both the left and the right view of the fish to determine the individual.
"""
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data import get_images, get_embeddings
from config import IMG_SIZE, DESCRIPTOR_SIZE
from metrics import map_at_k, mean_accuracy_at_k
import models
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from scipy.spatial import distance

# Configs
DISTANCE_METRIC = 'euclidean'
IGNORE_NEW_OBSERVATIONS = True


def load_dataset(dir, classes):
    X_train, y_train = get_images(os.path.join(DATASET_DIR, "train"), classes, IMG_SIZE, direction_labels=False)
    X_train = np.array(X_train)
    X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
    y_train = np.array(y_train)

    X_test, y_test = get_images(os.path.join(DATASET_DIR, "test"), classes, IMG_SIZE, direction_labels=False)
    X_test = np.array(X_test)
    X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)
    y_test = np.array(y_test)

    filenames = glob.glob(os.path.join(DATASET_DIR, "test", "*.jpg"))
    filenames = [os.path.basename(filename) for filename in filenames]

    drop_indices = []
    # Set y to -1 if record does not exist in the training set
    for i, y_test_i in enumerate(y_test):
        if y_test_i not in y_train:
            y_test[i] = -1
            drop_indices.append(i)

    if IGNORE_NEW_OBSERVATIONS:
        X_test = np.delete(X_test, drop_indices, axis=0)
        y_test = np.delete(y_test, drop_indices, axis=0)
        filenames = np.delete(filenames, drop_indices, axis=0)

    X_embeddings, y_embeddings = get_embeddings(left_model, X_train, y_train)
    columns = ["x" + str(i) for i in range(DESCRIPTOR_SIZE)] + ["y"]
    df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
    df = df.astype({"y": int})
    return df, X_train, y_train, X_test, y_test


def predict(df, X_test):
    embeddings = left_model.predict(X_test)
    y_list = []
    for i, embedding in enumerate(embeddings):
        if DISTANCE_METRIC == 'euclidean':
            df['distance'] = np.linalg.norm(df.iloc[:, :DESCRIPTOR_SIZE].sub(embedding), axis=1)
        elif DISTANCE_METRIC == 'cosine':
            df['distance'] = df.apply(lambda row: distance.cosine(embedding, row[:DESCRIPTOR_SIZE]), axis=1)
        df_sorted = df.copy()
        df_sorted = df_sorted.sort_values(by='distance', ignore_index=True)
        k_best = df_sorted['y'].tolist()
        y_list.append(k_best)
    return np.array(y_list)


def rank(y, predictions):
    rank = list(predictions).index(y)
    return rank


left_model = models.triplet_network_ohnm
left_model.load_weights('./models/left')

DATASET_DIR = os.path.join('..', 'data', 'dataset', 'cropped_head', 'direction_left')
with open("../data/classes_pairs.txt") as file:
    classes = [line.strip() for line in file]

df_left, X_train_left, y_train_left, X_test_left, y_test_left = load_dataset(DATASET_DIR, classes)
predictions_left = predict(df_left, X_test_left)

right_model = models.triplet_network_ohnm
right_model.load_weights('./models/right')

DATASET_DIR = os.path.join('..', 'data', 'dataset', 'cropped_head', 'direction_right')
with open("../data/classes_pairs.txt") as file:
    classes = [line.strip() for line in file]

df_right, X_train_right, y_train_right, X_test_right, y_test_right = load_dataset(DATASET_DIR, classes)
predictions_right = predict(df_right, X_test_right)

y_unique = list(np.unique(y_train_right))

# TODO: Må endre på metoden her. Denne krever at man vet y_true.

ranks_right = []
for predictions in predictions_right:
    r_list = []
    for y in y_unique:
        r = rank(y, predictions)
        r_list.append(r)
    ranks_right.append(r_list)

ranks_left = []
for predictions in predictions_left:
    r_list = []
    for y in y_unique:
        r = rank(y, predictions)
        r_list.append(r)
    ranks_left.append(r_list)

y_list = []
for ranks in zip(ranks_left, ranks_right):
    averages = {}
    for i in range(len(ranks[0])):
        avg = (ranks[0][i] + ranks[1][i])/2
        averages[y_unique[i]] = avg
    averages = {k: v for k, v in sorted(averages.items(), key=lambda item: item[1])}
    y_list.append(list(averages.keys()))

first_predictions = [y[0] for y in y_list]

accuracy = accuracy_score(y_test_right, first_predictions)
f1 = f1_score(y_test_right, first_predictions, average='macro')
mAcc_5 = mean_accuracy_at_k(y_test_right, y_list, 5)
mAP_5 = map_at_k(y_test_right, y_list, 5)
print("Accuracy: ", accuracy, "\nmAcc@5: ", mAcc_5, "\nMAP@5: ", mAP_5,  "\nF1: ", f1)

res_at_k = []
k_max = len(df_right) + 1
ks = list(range(1, k_max))
for k in ks:
    res = mean_accuracy_at_k(y_test_right, y_list, k)
    res_at_k.append(res)

plt.plot(ks, res_at_k)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

print("AUC: ", np.sum(res_at_k) / k_max)