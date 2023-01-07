"""
A simple CNN classifier that classifies individuals from images.
"""
import const
from data import load_dataset
import models

DATASET_FLAVOUR = const.FISH_HEAD

X_train, y_train, X_test, y_test, classes = load_dataset(DATASET_FLAVOUR, preprocess=True)

model = models.cnn_classifier
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16)

model.evaluate(X_test, y_test)

model.save("../models/cnn")