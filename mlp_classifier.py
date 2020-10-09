"""
A multilayer perceptron classifier for embeddings.
"""

import models
from data import get_images, IMG_SIZE, get_embeddings
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

DATASET_DIR = "../data/dataset_condensed/cropped_head/direction"

SAVE_EMBEDDINGS = False
LOAD_EMBEDDINGS = True

if SAVE_EMBEDDINGS:

    with open("../data/classes_condensed.txt") as file:
        classes = [line.strip() for line in file]

    X, y = get_images(DATASET_DIR, classes, IMG_SIZE)

    X = np.array(X)
    X = tf.keras.applications.inception_v3.preprocess_input(X)
    y = np.array(y)

    embedding_model = models.embedding_network_contrastive
    embedding_model.load_weights('./models/embedding')
    X_embeddings, y_embeddings = get_embeddings(embedding_model, X, y)

    columns = ["x" + str(i) for i in range(64)] + ["y"]
    df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
    df = df.astype({"y": int})
    df.to_csv("mlp_embeddings.csv", index=False)

if LOAD_EMBEDDINGS:

    df = pd.read_csv("mlp_embeddings.csv")
    X_embeddings = df.iloc[:, :-1].to_numpy()
    y_embeddings = df.iloc[:, -1].to_numpy()

classification_model = models.embedding_mlp
classification_model.compile(optimizer='adam',
                                  loss='sparse_categorical_crossentropy',
                                  metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y_embeddings, test_size=0.3, shuffle=True)

classification_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=2)
