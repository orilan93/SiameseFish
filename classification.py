"""
A collection of classifiers that uses the same interface so that they can be easily switched.
"""

import models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import tensorflow as tf
from data import get_embeddings

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

EMBED_IMAGES = True
SAVE_CSV = False
LOAD_CSV = False


class Classifier:
    def predict(self, x):
        pass

    def train(self, x, y):
        pass

    def evaluate(self, x, y):
        pass


class CNNClassifier(Classifier):
    def __init__(self):
        self.model = models.vgg16_classifier

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def predict(self, x):
        prediction = self.model.predict(x)
        return np.argmax(prediction, axis=1)

    def evaluate(self, x, y):
        self.model.evaluate(x, y, verbose=2)

    def train(self, x, y=None):
        if y is None:
            self.model.fit(x, epochs=10, steps_per_epoch=50)
        else:
            self.model.fit(x, y, epochs=100, batch_size=4)


class EmbeddingDistanceClassifier(Classifier):
    def __init__(self, threshold):
        # self.df = pd.read_csv("embeddings5.csv")
        # self.df_mean = self.df.groupby("y").mean()

        self.embedding_model = models.triplet_network_ohnm
        self.embedding_model.load_weights('./models/triplet_ohnm')
        self.threshold = threshold

    def predict(self, x):
        embeddings = self.embedding_model.predict(x)
        # self.df_mean['similarity'] = self.df_mean.apply(lambda row: array_distance(row[:32].to_numpy(), embedding), axis=1)
        y_list = []
        for embedding in embeddings:
            # self.df_mean['similarity'] = np.linalg.norm(self.df_mean.iloc[:, :32].sub(embedding), axis=1)
            self.df['distance'] = np.linalg.norm(self.df.iloc[:, :128].sub(embedding), axis=1)
            # tsorted = self.df.sort_values(by='distance')
            # y = self.df_mean['similarity'].idxmin()
            idx = self.df['distance'].idxmin()
            # y = self.df["y"][idx]
            distance = self.df['distance'][idx]
            if distance < self.threshold:
                y = self.df["y"][idx]
            else:
                y = -1
            y_list.append(y)
        return y_list

    def evaluate(self, x, y):
        prediction = self.predict(x)
        new_count = prediction.count(-1)
        accuracy = accuracy_score(y, prediction)
        f1 = f1_score(y, prediction, average='macro')
        print("Accuracy: ", accuracy, "\tF1: ", f1, "\t New: ", new_count)
        return accuracy

    def train(self, x, y=None):
        if EMBED_IMAGES:
            X_embeddings, y_embeddings = get_embeddings(self.embedding_model, x, y)
            columns = ["x" + str(i) for i in range(128)] + ["y"]
            self.df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
            self.df = self.df.astype({"y": int})
        if SAVE_CSV:
            self.df.to_csv("embeddings6.csv", index=False)
        if LOAD_CSV:
            self.df = pd.read_csv("embeddings6.csv")


class EmbeddingSVCClassifier(Classifier):
    def __init__(self):
        self.embedding_model = models.triplet_network_ohnm
        self.embedding_model.load_weights('./models/triplet_ohnm')
        self.classification_model = SVC(kernel='linear', probability=True, C=1.0)
        # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
        # self.classification_model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=2)

    def predict(self, x):
        embedding = self.embedding_model.predict(x)
        prediction = self.classification_model.predict(embedding)
        return prediction

    def evaluate(self, x, y):
        embedding = self.embedding_model.predict(x)
        predictions = self.classification_model.predict(embedding)
        print(accuracy_score(predictions, y))

    def train(self, x, y=None):
        if y is None:
            raise NotImplemented
        else:
            if EMBED_IMAGES:
                X_embeddings, y_embeddings = get_embeddings(self.embedding_model, x, y)
            if SAVE_CSV:
                columns = ["x" + str(i) for i in range(32)] + ["y"]
                df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
                df = df.astype({"y": int})
                df.to_csv("embeddings5.csv", index=False)
            if LOAD_CSV:
                df = pd.read_csv("embeddings5.csv")
                X_embeddings = df.iloc[:, :-1].to_numpy()
                y_embeddings = df.iloc[:, -1].to_numpy()
            self.classification_model.fit(X_embeddings, y_embeddings)
            # print(self.classification_model.best_params_)


class EmbeddingKNNClassifier(Classifier):
    def __init__(self, k=1):
        self.embedding_model = models.triplet_network_ohnm
        self.embedding_model.load_weights('./models/triplet_ohnm')
        self.classification_model = KNeighborsClassifier(k)
        pass

    def predict(self, x):
        embedding = self.embedding_model.predict(x)
        prediction = self.classification_model.predict(embedding)
        return prediction

    def evaluate(self, x, y):
        embedding = self.embedding_model.predict(x)
        predictions = self.classification_model.predict(embedding)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='macro')
        print("Accuracy: ", accuracy, "\tF1: ", f1)
        return accuracy

    def train(self, x, y=None):
        if y is None:
            raise NotImplemented
        else:
            if EMBED_IMAGES:
                X_embeddings, y_embeddings = get_embeddings(self.embedding_model, x, y)
            if SAVE_CSV:
                columns = ["x" + str(i) for i in range(64)] + ["y"]
                df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
                df = df.astype({"y": int})
                df.to_csv("embeddings7.csv", index=False)
            if LOAD_CSV:
                df = pd.read_csv("embeddings7.csv")
                X_embeddings = df.iloc[:, :-1].to_numpy()
                y_embeddings = df.iloc[:, -1].to_numpy()
            self.classification_model.fit(X_embeddings, y_embeddings)


class EmbeddingMLPClassifier(Classifier):
    def __init__(self):
        self.embedding_model = models.embedding_network_contrastive
        self.embedding_model.load_weights('./models/embedding')
        self.classification_model = models.embedding_mlp
        self.classification_model.compile(optimizer='adam',
                                          loss='sparse_categorical_crossentropy',
                                          metrics=['accuracy'])
        pass

    def predict(self, x):
        embedding = self.embedding_model.predict(x)
        prediction = self.classification_model.predict(embedding)
        prediction = np.argmax(prediction, axis=1)
        return prediction

    def evaluate(self, x, y):
        embedding = self.embedding_model.predict(x)
        self.classification_model.evaluate(embedding, y)

    def train(self, x, y=None):
        if y is None:
            raise NotImplemented
        else:
            if EMBED_IMAGES:
                X_embeddings, y_embeddings = get_embeddings(self.embedding_model, x, y)
            if LOAD_CSV:
                df = pd.read_csv("embeddings5.csv")
                X_embeddings = df.iloc[:, :-1].to_numpy()
                y_embeddings = df.iloc[:, -1].to_numpy()
            self.classification_model.fit(X_embeddings, np.array(y_embeddings), epochs=500, batch_size=256)


# (x_train, y_train), (x_test, y_test) = get_mnist()
#
# embedding_model = embedding_network
# embedding_model.load_weights('./models/embedding_network')
# X_all = np.concatenate((x_train, x_test), axis=0)
# y_all = np.concatenate((y_train, y_test), axis=0)
# embeddings = get_embeddings(embedding_model, X_all, y_all)
# pickle.dump(embeddings, open("embeddings.p", "wb"))
#
# X_embeddings = np.concatenate(list(embeddings.values()))
# y_embeddings = [[k]*len(embeddings[k]) for k in embeddings.keys()]
# y_embeddings = [item for sublist in y_embeddings for item in sublist]
#
# classification_model = SVC(kernel='linear', probability=True)
#
# X_embeddings_train, X_embeddings_test, y_embeddings_train, y_embeddings_test =\
#     train_test_split(X_embeddings, y_embeddings, test_size=0.1, shuffle=True)
#
# classification_model.fit(X_embeddings_train, y_embeddings_train)
#
# predictions = classification_model.predict(X_embeddings_test)
# accuracy = accuracy_score(predictions, y_embeddings_test)
# print(accuracy)
#
# prediction = classification_model.predict(X_embeddings_test[:4])
#
# fig, ax = plt.subplots(nrows=4, ncols=2)
# for i, row in enumerate(ax):
#     row[0].imshow(X_embeddings_test[i].reshape(28, 28))
#     row[1].text(0, 0, str(prediction[i]))
#
# plt.show()
