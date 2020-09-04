"""
A collection of performance metrics and loss functions.
"""

import tensorflow as tf
import numpy as np

from config import DESCRIPTOR_SIZE
from utils import groupby_y
import scipy.spatial.distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def triplet_loss(y_true, y_pred, alpha=1.0):
    n_features = DESCRIPTOR_SIZE
    anchor, positive, negative = tf.unstack(tf.reshape(y_pred, (-1, 3, n_features)), num=3, axis=1)

    positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
    loss = tf.maximum(basic_loss, 0.0)
    loss = tf.reduce_mean(loss, 0)

    return loss


def precision_at_k(y_true, predictions, k):
    """Returns the precision at k. This is a simplified version of precision and assumes binary classification."""

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
        prec = precision_at_k(y_true[i], pred, k)
        mAP.append(prec)
    mAP = np.mean(mAP)
    return mAP


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


def above_threshold(y_true, y_pred):
    """Metrics that is true if a point is above 0.5 and false otherwise."""
    #return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < 0.5, y_true.dtype)), tf.float32))
    return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred > 0.5, y_true.dtype)), tf.float32))


def nearest_neighbor_classes(model, X_train, X_test, y_train):
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
            cls = y_train[n]  # Lookup the actual class of the point in the support set TODO: Verify if this is correct
            sample.append((cls, d))

        # Naively adds a new observation with distance 0.5 to the anchor.
        # This means that it will always predict that it's a new observation if there is no points within 0.5.
        #sample.append((-1, 1.0))

        # Sort based on distance
        sample.sort(key=lambda x: x[1])

        sample = [s[0] for s in sample]
        predictions.append(sample)

    return predictions


def map_metric(model, X_train, y_train, X_test, y_test):
    """Splits the dataset and evaluates the MAP @1 and @5 metrics."""

    #X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.4)

    for i, y_test_i in enumerate(y_test):
        if y_test_i not in y_train:
            y_test[i] = -1

    predictions = nearest_neighbor_classes(model, X_train, X_test, y_train)

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