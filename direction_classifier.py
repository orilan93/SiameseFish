"""
Classifier for the direction the fish is facing in the image.
"""

import os
from PIL import Image
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

from models import direction_detector
from data import resize_padding
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from config import IMG_SIZE

#DATA_DIR = os.path.join("..", "data")
#DATASET_DIR = os.path.join(DATA_DIR, "dataset", "cropped_head")
#DIRECTION_FILE = os.path.join(DATA_DIR, "direction.txt")
DATA_DIR = os.path.join("..", "data", "quad", "first_batch")
DATASET_DIR = os.path.join(DATA_DIR, "cropped")
DIRECTION_FILE = os.path.join(DATA_DIR, "direction.txt")
RETRAIN = False

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
    if os.path.isfile(image_path):
        pil_img = Image.open(image_path)
        pil_img = resize_padding(pil_img, IMG_SIZE)

        X.append(np.array(pil_img))
        y.append(value)

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

keras_augmentor = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=50.0,
    brightness_range=[0.9, 1.1],
    # preprocessing_function=flip_channels,
    # preprocessing_function=to_greyscale
    # vertical_flip=True,
    # horizontal_flip=True
)
X_train = keras_augmentor.flow(X_train, batch_size=len(X_train), shuffle=False)
X_train = next(X_train)

X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)


model = direction_detector

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

if RETRAIN:
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

    model.save_weights('./models/direction_quad')
else:
    model.load_weights('./models/direction_quad')

model.evaluate(X_test, y_test)

predictions = model.predict(X_test[8:16])

fig, ax = plt.subplots(nrows=4, ncols=4)
for i, row in enumerate(ax):
    row[0].imshow(X_test[i*2].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[1].text(0, 0, "left" if predictions[i*2] < 0.5 else "right")
    row[2].imshow(X_test[i*2 + 1].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[3].text(0, 0, "left" if predictions[i*2 + 1] < 0.5 else "right")

plt.show()
