"""
Classifies an embedding as it's closest neighbor in the support set.
If the distance to closest neighbor is more than a threshold it is classified as a new observation.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data import get_images, get_omniglot, extract_pattern
from config import IMG_SIZE, DESCRIPTOR_SIZE
from metrics import map_at_k
import models
from classification import get_embeddings
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Datasets
FISH = 0
OMNIGLOT = 1
FRUIT = 2

# Dataset flavours
FISH_HEAD_DIRECTION = 0
FISH_MALES = 1
FISH_HEAD_LEFT = 2
FISH_WHOLE = 3

# Configs
BATCH_SIZE = 5
THRESHOLD = 10.0
RETRAIN = True
DEMO = False
USE_DATASET = FISH
DATASET_FLAVOUR = FISH_HEAD_DIRECTION
IGNORE_NEW_OBSERVATIONS = True

model = models.triplet_network_ohnm
model.load_weights('./models/embedding')

if USE_DATASET == FISH:
    if DATASET_FLAVOUR == FISH_HEAD_DIRECTION:
        DATASET_DIR = "../data/dataset_condensed/cropped_head/direction"
        with open("../data/classes_condensed_head.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_MALES:
        DATASET_DIR = "../data/dataset_condensed/males/cropped_head/direction"
        with open("../data/classes_condensed_males.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_HEAD_LEFT:
        DATASET_DIR = "../data/dataset_condensed/cropped_head/direction_left"
        with open("../data/classes_condensed_head_left.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_WHOLE:
        DATASET_DIR = "../data/dataset_condensed/cropped/direction"
        with open("../data/classes_condensed_direction.txt") as file:
            classes = [line.strip() for line in file]

    X_train, y_train = get_images(DATASET_DIR + "\\train", classes, IMG_SIZE)
    # for i, x in enumerate(X_train):
    #     X_train[i] = np.tile(extract_pattern(x), (1, 1, 3))
    X_train = np.array(X_train)
    X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
    y_train = np.array(y_train)

    X_test, y_test = get_images(DATASET_DIR + "\\test", classes, IMG_SIZE)
    # for i, x in enumerate(X_test):
    #     X_train[i] = np.tile(extract_pattern(x), (1, 1, 3))
    X_test = np.array(X_test)
    X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)
    y_test = np.array(y_test)

if USE_DATASET == OMNIGLOT:
    X, y = get_omniglot()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

if USE_DATASET == FRUIT:
    DATASET_DIR = "../data/fruit"
    with open("../data/classes_fruit.txt") as file:
        classes = [line.strip() for line in file]
    X, y = get_images(DATASET_DIR, classes, IMG_SIZE)
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

drop_indices = []
# Set y to -1 if record does not exist in the training set
for i, y_test_i in enumerate(y_test):
    if y_test_i not in y_train:
        y_test[i] = -1
        drop_indices.append(i)


if IGNORE_NEW_OBSERVATIONS:
    X_test = np.delete(X_test, drop_indices, axis=0)
    y_test = np.delete(y_test, drop_indices, axis=0)

#X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

print("test set size: ", str(len(y_test)))
unique, counts = np.unique(y_test, return_counts=True)
new_observations = dict(zip(unique, counts))
new_observations = new_observations[-1] if -1 in new_observations else 0
print("y_test new observations: ", str(new_observations))

X_embeddings, y_embeddings = get_embeddings(model, X_train, y_train)
columns = ["x" + str(i) for i in range(DESCRIPTOR_SIZE)] + ["y"]
df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
df = df.astype({"y": int})


def predict(x, y_true):
    embeddings = model.predict(x)
    y_list = []
    distances = []
    for i, embedding in enumerate(embeddings):
        df['distance'] = np.linalg.norm(df.iloc[:, :DESCRIPTOR_SIZE].sub(embedding), axis=1)
        df_sorted = df.sort_values(by='distance', ignore_index=True)
        y_true_i = y_true[i]
        if y_true_i != -1:
            sorted_y_list = list(df_sorted.y.values)
            idx_true = sorted_y_list.index(y_true_i)
            true_distance = df_sorted['distance'][idx_true]
            distances.append(true_distance)
        idx = df['distance'].idxmin()
        distance = df['distance'][idx]
        if distance < THRESHOLD:
            y_pred = df["y"][idx]
        else:
            y_pred = -1
        y_list.append(y_pred)
    return y_list, distances


def predict_k(x, y_true, k=1):
    embeddings = model.predict(x)
    y_list = []
    for i, embedding in enumerate(embeddings):
        df['distance'] = np.linalg.norm(df.iloc[:, :DESCRIPTOR_SIZE].sub(embedding), axis=1)
        df_sorted = df.append({'y': -1, 'distance': THRESHOLD}, ignore_index=True)
        df_sorted = df_sorted.sort_values(by='distance', ignore_index=True)
        k_best = df_sorted['y'][:k].tolist()
        y_list.append(k_best)
    return np.array(y_list)


def evaluate(x, y, report=False):
    prediction, distances = predict(x, y)
    new_count = prediction.count(-1)
    accuracy = accuracy_score(y, prediction)
    f1 = f1_score(y, prediction, average='macro')
    mean_error = np.array(distances).mean()
    print("Accuracy: ", accuracy, "\tF1: ", f1, "\tMean Error:", str(mean_error), "\t New: ", new_count)
    if report:
        print(classification_report(y, prediction))#, classes))
    return accuracy, f1


def evaluate2(x, y):
    predictions = predict_k(x, y, k=5)
    new_count = list(predictions[:, 0]).count(-1)
    accuracy = accuracy_score(y, predictions[:, 0])
    f1 = f1_score(y, predictions[:, 0], average='macro')
    mAP_5 = map_at_k(y_test, predictions, 5)
    print("MAP@5: ", mAP_5, "\tAccuracy: ", accuracy, "\tF1: ", f1, "\t New: ", new_count)


# best_f1 = 0
# best_t = -1
# #for t in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.1]:
# for t in [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
#     print("t = ", t)
#     threshold = t
#     accuracy, f1 = evaluate(X_valid, y_valid)
#     if f1 > best_f1:
#         best_f1 = f1
#         best_t = t
#     print("======================================")
#
# print("best t: ", best_t)

# threshold = best_t
# accuracy, f1 = evaluate(X_test, y_test, report=False)
evaluate2(X_test, y_test)

# from sklearn.neighbors import KNeighborsClassifier
# knn_model = KNeighborsClassifier(1)
# knn_model.fit(X_embeddings, y_embeddings)
# embeddings = model.predict(X_test)
# predictions = knn_model.predict(embeddings)
# accuracy = accuracy_score(y_test, predictions)
# print(accuracy)
#print(classification_report(y_test, predictions))

# prediction, distances = predict(X_test[:4], y_test[:4])
#
# fig, ax = plt.subplots(nrows=4, ncols=2)
# for i, row in enumerate(ax):
#    row[0].imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
#    pred_class = -1 if prediction[i] == -1 else classes[prediction[i]]
#    true_class = -1 if prediction[i] == -1 else classes[y_test[i]]
#    row[1].text(0, 0, "Predicted: " + str(pred_class))
#    row[1].text(0, 0.5, "Ground truth: " + str(true_class)) # Check if y is -1
#
# plt.show()