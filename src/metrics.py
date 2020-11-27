"""
A collection of performance metrics and loss functions.
"""

import tensorflow as tf
import numpy as np
from config import DESCRIPTOR_SIZE
from utils import groupby_y
import scipy.spatial.distance
from sklearn.neighbors import NearestNeighbors


def triplet_loss(y_true, y_pred, alpha=1.0):
    """Calculates triplet loss for siamese network."""
    n_features = DESCRIPTOR_SIZE
    anchor, positive, negative = tf.unstack(tf.reshape(y_pred, (-1, 3, n_features)), num=3, axis=1)

    positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
    loss = tf.maximum(basic_loss, 0.0)
    loss = tf.reduce_mean(loss, 0)

    return loss


def ntxent_loss(y_true, y_pred):
    """Calculates NT-Xent loss."""
    return add_contrastive_loss(y_pred)[0]


LARGE_NUM = 1e9


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         weights=1.0):
    """Compute loss for model.
    -- From SimCLR repository.
    Args:
    hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.
    Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    # Gather hidden1/hidden2 across replicas and create local labels
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.compat.v1.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
    loss_b = tf.compat.v1.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
    loss = loss_a + loss_b

    return loss, logits_ab, labels


def average_precision_at_k(y_true, predictions, k):
    """Returns the average precision at k.
    This is a simplified version of average precision and assumes binary classification."""

    for i, pred in enumerate(predictions):
        if i >= k:
            return 0
        if pred == y_true:
            return 1.0 / (i + 1)
    return 0


def map_at_k(y_true, predictions, k):
    """Returns the mean average precision at k."""

    mAP = []
    for i, pred in enumerate(predictions):
        prec = average_precision_at_k(y_true[i], pred, k)
        mAP.append(prec)
    mAP = np.mean(mAP)
    return mAP


def accuracy_at_k(y_true, predictions, k):
    """Returns the accuracy at k. Returns 1 if the correct prediction is within the k best predictions."""

    for i, pred in enumerate(predictions):
        if i >= k:
            return 0
        if pred == y_true:
            return 1
    return 0


def mean_accuracy_at_k(y_true, predictions, k):
    """Modified mAP that gives an accuracy of 1 if the correct prediction is within the k best predictions."""
    res = []
    for i, pred in enumerate(predictions):
        acc = accuracy_at_k(y_true[i], pred, k)
        res.append(acc)
    return np.mean(res)


def above_threshold(y_true, y_pred):
    """Mean of number of prediction above 0.5."""
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred > 0.5, y_true.dtype)), tf.float32))


def nearest_neighbor_classes(model, X_train, X_test, y_train, ignore_new=False):
    """Returns the classes of the nearest points to the anchors sorted by distance."""

    # Setup nearest neighbor algorithm
    neigh = NearestNeighbors(n_neighbors=6)

    # Get training embeddings
    embeddings_train = model.predict(X_train)

    # Add support set
    neigh.fit(embeddings_train)

    # Get testing embeddings
    embeddings_test = model.predict(X_test)

    # Gets the closest neighbors of each point for the test embeddings in the support set and their distance
    distances_test, neighbors_test = neigh.kneighbors(embeddings_test)
    distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()

    predictions = []
    for distances, neighbors in zip(distances_test, neighbors_test):  # Loop through test set
        sample = []

        for d, n in zip(distances, neighbors):  # Loop through nearest points
            cls = y_train[n]  # Lookup the actual class of the point in the support set
            sample.append((cls, d))

        # Naively adds a new observation with distance 0.5 to the anchor.
        # This means that it will always predict that it's a new observation if there is no points within 0.5.
        if not ignore_new:
            sample.append((-1, 1.0))

        # Sort based on distance
        sample.sort(key=lambda x: x[1])

        sample = [s[0] for s in sample]
        predictions.append(sample)

    return predictions


def map_metric(model, X_train, y_train, X_test, y_test, ignore_new=False):
    """Evaluates the MAP @1 and @5 metrics."""

    drop_indices = []
    y_test = y_test.copy()
    for i, y_test_i in enumerate(y_test):
        if y_test_i not in y_train:
            y_test[i] = -1
            drop_indices.append(i)

    if ignore_new:
        X_test = np.delete(X_test, drop_indices, axis=0)
        y_test = np.delete(y_test, drop_indices, axis=0)

    X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
    X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)

    predictions = nearest_neighbor_classes(model, X_train, X_test, y_train, ignore_new=ignore_new)

    mAP_5 = map_at_k(y_test, predictions, 5)
    mAP_1 = map_at_k(y_test, predictions, 1)
    print(" - MAP@1: ", mAP_1, " - MAP@5: ", mAP_5)


def closeness_metric(model, X, y):
    """Prints the average of the pairwise distances between the points in the classes."""
    y_pred = model.predict(X)
    y_pred_grouped = groupby_y(y_pred, y)
    avg_d = []
    for g in y_pred_grouped:
        d = scipy.spatial.distance.pdist(np.array(g), metric='euclidean')
        avg_d.append(np.sum(d))
    print(" - Closeness: ", np.mean(avg_d))
