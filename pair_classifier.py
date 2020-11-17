"""
A classifier that uses both the left and the right view of the fish to determine the individual.
"""
import os
import numpy as np
import tensorflow as tf
from data import get_images, get_embeddings
from config import IMG_SIZE, DESCRIPTOR_SIZE
import models
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, f1_score
import random
import metrics
from predictions import show_accuracy_curve

# Configs
DISTANCE_METRIC = 'euclidean'
IGNORE_NEW_OBSERVATIONS = True

DATASET_DIR_LEFT = os.path.join('..', 'data', 'merged', 'head', 'Oct', 'left')
DATASET_DIR_RIGHT = os.path.join('..', 'data', 'merged', 'head', 'Oct', 'right')
# DATASET_DIR_LEFT = os.path.join('..', 'data', 'dataset', 'cropped_head', 'direction_left')
# DATASET_DIR_RIGHT = os.path.join('..', 'data', 'dataset', 'cropped_head', 'direction_right')
with open("../data/classes.txt") as file:
    classes = [line.strip() for line in file]


def load_dataset(dir, classes):
    """Loads a dataset from a directory and return train/test split."""

    X_train, y_train = get_images(os.path.join(dir, "train"), classes, IMG_SIZE, direction_labels=False)
    X_train = np.array(X_train)
    X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
    y_train = np.array(y_train)

    X_test, y_test = get_images(os.path.join(dir, "test"), classes, IMG_SIZE, direction_labels=False)
    X_test = np.array(X_test)
    X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def remove_from_test(x_test_left, y_test_left, x_test_right, y_test_right, y_train_left, y_train_right):
    """Remove from test set, if the class doesn't exists in the train set."""

    drop_indices = []
    for i, y_test_i in enumerate(y_test_left):
        if y_test_i not in y_train_left:
            y_test_left[i] = -1
            drop_indices.append(i)

    for i, y_test_i in enumerate(y_test_right):
        if y_test_i not in y_train_right:
            y_test_right[i] = -1
            drop_indices.append(i)

    if IGNORE_NEW_OBSERVATIONS:
        x_test_left = np.delete(x_test_left, drop_indices, axis=0)
        y_test_left = np.delete(y_test_left, drop_indices, axis=0)

        x_test_right = np.delete(x_test_right, drop_indices, axis=0)
        y_test_right = np.delete(y_test_right, drop_indices, axis=0)

    return x_test_left, y_test_left, x_test_right, y_test_right


def create_dataframe(model, x_train, y_train):
    """Creates a dataframe from a support dataset."""
    X_embeddings, y_embeddings = get_embeddings(model, x_train, y_train)
    columns = ["x" + str(i) for i in range(DESCRIPTOR_SIZE)] + ["y"]
    df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
    df = df.astype({"y": int})
    return df


def predict(model, df, x_test):
    """Predicts a batch of images given an embedding model and support set."""
    embeddings = model.predict(x_test)
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
    """Gets the position of y in the predictions list, or -1 if it does not exist."""
    if y in predictions:
        rank = list(predictions).index(y)
    else:
        rank = -1
    return rank


def get_ranks_from_predictions(predictions_list, y_values):
    """Gets the ranks of each y in y_values for each set of predictions in predictions_list."""
    ranks_list = []
    for predictions in predictions_list:
        r_list = []
        for y in y_values:
            r = rank(y, predictions)
            r_list.append(r)
        ranks_list.append(r_list)
    return ranks_list


def pair_predict(pred_left, pred_right, classes):
    """Predicts individuals given a batch of predictions of both sides.
    If left predicts a certain class at first attempt and right predicts it
    at the third attempt, it will now be seen as the second most probable class."""

    # Gets a list of all possible y values. These will be used for
    y_values = list(range(len(classes)))

    # Gets the ranks of all the predictions
    ranks_left = get_ranks_from_predictions(pred_left, y_values)
    ranks_right = get_ranks_from_predictions(pred_right, y_values)

    y_list = []
    for ranks in zip(ranks_left, ranks_right):  # For the set of ranks in each prediction
        averages = {}
        for i in range(len(ranks[0])):
            if ranks[0][i] == -1:
                # These two cases should probably not happen unless you constrain the number of attempts
                # or y_values contain values that are outside of the prediction range.
                avg = ranks[1][i]
            elif ranks[1][i] == -1:
                avg = ranks[0][i]
            else:
                avg = (ranks[0][i] + ranks[1][i]) / 2
            averages[y_values[i]] = avg

        # Calculate a new proposal for n predictions using the average
        # of what is being predicted by the left and right side.
        averages = {k: v for k, v in sorted(averages.items(), key=lambda item: (item[1] < 0, item[1]))}
        y_list.append(list(averages.keys()))

    return y_list


# Load models
left_model = models.triplet_network_ohnm
left_model.load_weights('./models/left')
# Cloning is necessary or else it will just overwrite the other model
right_model = tf.keras.models.clone_model(models.triplet_network_ohnm)
right_model.load_weights('./models/right')

# Load datasets
X_train_left, y_train_left, X_test_left, y_test_left = load_dataset(DATASET_DIR_LEFT, classes)
X_train_right, y_train_right, X_test_right, y_test_right = load_dataset(DATASET_DIR_RIGHT, classes)

# Remove from test set if class doesn't exist in train set
X_test_left, y_test_left, X_test_right, y_test_right = remove_from_test(
    X_test_left, y_test_left, X_test_right, y_test_right, y_train_left, y_train_right)

print("y_test_left size: {}".format(len(y_test_left)))
print("y_test_right size: {}".format(len(y_test_right)))

# Create dataframes of the support set
df_left = create_dataframe(left_model, X_train_left, y_train_left)
df_right = create_dataframe(right_model, X_train_right, y_train_right)

# Sanity check that the left and right dataset is actually paired
# assert ((y_train_left == y_train_right).all())
assert ((y_test_left == y_test_right).all())
y_test = y_test_right

print("Support set size: {}".format(len(df_left)))

# Get predictions for both sides
predictions_left = predict(left_model, df_left, X_test_left)
predictions_right = predict(right_model, df_right, X_test_right)

# Get accuracies for individual sides
accuracy_left = accuracy_score(y_test, [y[0] for y in predictions_left])
accuracy_right = accuracy_score(y_test, [y[0] for y in predictions_right])
print("Accuracy left: {}, Accuracy right: {}".format(accuracy_left, accuracy_right))

# Get predictions when using both sides
y_pred = np.array(pair_predict(predictions_left, predictions_right, classes))

# Slice away unused values
y_pred = y_pred[:, :len(df_left)]
assert (not (y_pred == -1).any())
print("Prediction set shape: {}".format(y_pred.shape))
first_predictions = y_pred[:, 0]

# TODO: Handle rank=-1 cases better

# Generate report
accuracy = accuracy_score(y_test, first_predictions)
f1 = f1_score(y_test, first_predictions, average='macro')
mAcc_5 = metrics.mean_accuracy_at_k(y_test, y_pred, 5)
mAP_5 = metrics.map_at_k(y_test, y_pred, 5)
print("Accuracy: ", accuracy, "\nmAcc@5: ", mAcc_5, "\nMAP@5: ", mAP_5, "\nF1: ", f1)

auc = show_accuracy_curve(y_pred, y_test, return_auc=True)
print("AUC: ", auc)

print("\nTesting some random samples from test set.")

for i in range(4):

    r = random.randint(0, len(y_test_left))

    prediction_right = predict(right_model, df_right, np.expand_dims(X_test_right[r], 0))
    prediction_left = predict(left_model, df_left, np.expand_dims(X_test_left[r], 0))

    y_pred = pair_predict(prediction_right, prediction_left, classes)[0]
    gt = classes[y_test_left[r]]
    first_guess = classes[y_pred[0]]

    print("\nPredicted {}, and ground truth is {}".format(first_guess, gt))

    if first_guess != gt:
        correct_attempt = rank(y_test_left[r], y_pred)
        print("However on the {}. attempt, {} was predicted.".format(correct_attempt + 1,
                                                                     classes[y_pred[correct_attempt]]))
