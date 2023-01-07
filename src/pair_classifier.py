"""
A classifier that uses both the left and the right view of the fish to determine the individual.
"""
import numpy as np
import tensorflow as tf
from data import get_embeddings, load_dataset
from config import DESCRIPTOR_SIZE
import models
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import random
import metrics
from predictions import show_accuracy_curve, predict_k
import const

# Configs
DISTANCE_METRIC = 'euclidean'
IGNORE_NEW_OBSERVATIONS = True
CLASSIFICATION_METHOD = 'rank'  # rank, distance


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


def rank(y, predictions):  # TODO: Duplicate code?
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


def pair_predict_distance(left_model, right_model, df_left, df_right, X_test_left, X_test_right):
    left_embeddings = left_model.predict(X_test_left)
    right_embeddings = right_model.predict(X_test_right)

    df = pd.concat([df_left, df_right])

    y_pred = []

    cnt = 0
    for left_emb, right_emb in zip(left_embeddings, right_embeddings):
        print(str(cnt) + "/" + str(len(left_emb)) + "\r")
        cnt+= 1

        y_list = []

        for i, ref in df.iterrows():
            dist_left = np.linalg.norm(ref.iloc[:DESCRIPTOR_SIZE].sub(left_emb))
            dist_right = np.linalg.norm(ref.iloc[:DESCRIPTOR_SIZE].sub(right_emb)) # TODO: OMG FEIL!!
            dist_sqr = (dist_left + dist_right)**2
            # TODO: må optimaliseres
            y_list.append((ref["y"], dist_sqr))

        y_list = sorted(y_list, key=lambda x: x[1])
        y_pred.append([tup[0] for tup in y_list])
    print("\n")
    return np.array(y_pred)


# Load models
left_model = models.triplet_network_ohnm
left_model.load_weights('../models/head_left')
# Cloning is necessary or else it will just overwrite the other model
right_model = tf.keras.models.clone_model(models.triplet_network_ohnm)
right_model.load_weights('../models/head_right')

# Load datasets
X_train_left, y_train_left, X_test_left, y_test_left, classes = load_dataset(const.FISH_HEAD_PAIR_LEFT, preprocess=True)
X_train_right, y_train_right, X_test_right, y_test_right, classes = load_dataset(const.FISH_HEAD_PAIR_RIGHT,
                                                                                 preprocess=True)

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


if CLASSIFICATION_METHOD == 'rank':

    # Get predictions for both sides
    # predictions_left = predict(left_model, df_left, X_test_left)
    # predictions_right = predict(right_model, df_right, X_test_right)
    predictions_left = predict_k(left_model, df_left, X_test_left, k=-1, predict_new=False)
    predictions_right = predict_k(right_model, df_right, X_test_right, k=-1, predict_new=False)

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

else:
    y_pred = pair_predict_distance(left_model, right_model, df_left, df_right, X_test_left, X_test_right)

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

    prediction_right = predict_k(right_model, df_right, np.expand_dims(X_test_right[r], 0))
    prediction_left = predict_k(left_model, df_left, np.expand_dims(X_test_left[r], 0))

    y_pred = pair_predict(prediction_right, prediction_left, classes)[0]
    gt = classes[y_test_left[r]]
    first_guess = classes[y_pred[0]]

    print("\nPredicted {}, and ground truth is {}".format(first_guess, gt))

    if first_guess != gt:
        correct_attempt = rank(y_test_left[r], y_pred)
        print("However on the {}. attempt, {} was predicted.".format(correct_attempt + 1,
                                                                     classes[y_pred[correct_attempt]]))
