"""
A simple CNN classifier that classifies an image directly into an individual.
"""

from data import get_images
from config import IMG_SIZE
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import models

DATASET_DIR = "../data/dataset/cropped_head/direction"

with open("../data/classes_direction.txt") as file:
    classes = [line.strip() for line in file]

X, y = get_images(DATASET_DIR, classes, IMG_SIZE)

X = np.array(X)
X = tf.keras.applications.inception_v3.preprocess_input(X)
y = np.array(y)

model = models.cnn_classifier
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)

model.evaluate(X_test, y_test)