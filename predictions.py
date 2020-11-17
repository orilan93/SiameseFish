"""
Prediction and evaluation functions.
"""

from scipy.spatial import distance
from sklearn.metrics import accuracy_score, f1_score
from config import DESCRIPTOR_SIZE
import metrics
import matplotlib.pyplot as plt
import numpy as np


def show_accuracy_curve(y_pred, y_test, return_auc=False):
    """Displays a curve that shows how many guesses are needed before a certain accuracy is achieved."""
    res_at_k = []
    k_max = y_pred.shape[1] + 1

    ks = list(range(1, k_max))
    for k in ks:
        res = metrics.mean_accuracy_at_k(y_test, y_pred, k)
        res_at_k.append(res)

    plt.plot(ks, res_at_k)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.ylim(-0.1, 1.1)
    plt.show()

    if return_auc:
        return np.sum(res_at_k) / k_max


def predict_k(model, df_embeddings, x_test, k=1, metric='euclidean', predict_new=True, threshold=0.5, exact_class=False,
              y_test=None, y_test_exact=None):
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
    mAP_5 = metrics.map_at_k(y_true, y_pred, 5)
    mAcc_5 = metrics.mean_accuracy_at_k(y_true, y_pred, 5)
    print("Accuracy: ", accuracy, "\nmAcc@5: ", mAcc_5, "\nMAP@5: ", mAP_5, "\nF1: ", f1, "\nNew: ", new_count)

    if accuracy_curve:
        auc = show_accuracy_curve(y_pred, y_true, return_auc=True)
        print("AUC: ", auc)