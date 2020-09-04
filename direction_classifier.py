"""
Classifier for the direction the fish is facing in the image.
"""

import os
from PIL import Image
import numpy as np
from models import direction_detector
from data import resize_padding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

DATA_DIR = "../data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_condensed/cropped")
DIRECTION_FILE = os.path.join(DATA_DIR, "direction_condensed.txt")
IMG_SIZE = 416 # TODO: Load from config
RETRAIN = True

directions = dict()

with open(DIRECTION_FILE) as file:
    for line in file:
        line = line.strip()
        row = line.split(",")
        directions[row[0]] = int(row[1])

X = []
y = []

for key, value in directions.items():
    image_path = os.path.join(DATASET_DIR, key)
    pil_img = Image.open(image_path)
    pil_img = resize_padding(pil_img, IMG_SIZE)

    X.append(np.array(pil_img))
    y.append(value)

X = np.array(X)
X = tf.keras.applications.inception_v3.preprocess_input(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = direction_detector

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

if RETRAIN:
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

    model.save_weights('./models/direction')
else:
    model.load_weights('./models/direction')

model.evaluate(X_test, y_test)

predictions = model.predict(X_test[:8])

fig, ax = plt.subplots(nrows=4, ncols=4)
for i, row in enumerate(ax):
    row[0].imshow(X_test[i*2].reshape(IMG_SIZE, IMG_SIZE, 3))
    row[1].text(0, 0, "left" if predictions[i*2] < 0.5 else "right")
    row[2].imshow(X_test[i*2 + 1].reshape(IMG_SIZE, IMG_SIZE, 3))
    row[3].text(0, 0, "left" if predictions[i*2 + 1] < 0.5 else "right")

plt.show()
