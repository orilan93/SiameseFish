"""
Network that learns good embeddings from images.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa

from config import IMG_SHAPE, IMG_SIZE
from data import define_ohnm_train_test_generators, get_omniglot, define_siamese_train_test_generators, \
    define_triplet_train_test_generators, get_images, flip_channels, define_ohnm_train_test_generators2, saliency_map, \
    get_heavy_augmentor, extract_pattern
from metrics import map_metric, above_threshold, triplet_loss
import models

# Network types
SIAMESE_BINARY = 0
SIAMESE_CONTRASTIVE = 1
SIAMESE_TRIPLET = 2
OHNM_TRIPLET = 3

# Datasets
FISH = 0
OMNIGLOT = 1
FRUIT = 2

# Dataset flavours
FISH_HEAD_DIRECTION = 0
FISH_MALES = 1
FISH_HEAD_LEFT = 2
FISH_WHOLE = 3

# Configs
RETRAIN = True
BATCH_SIZE = 16
NETWORK_TYPE = OHNM_TRIPLET
USE_DATASET = FISH
DATASET_FLAVOUR = FISH_HEAD_DIRECTION

# Runs through fine tuning steps specified by (unfreeze from layer, epochs, learning rate)
# fine_tuning = [(None, 10, 1e-3), (276, 10, 1e-4), (248, 80, 1e-4)]
# fine_tuning = [(None, 20, 1e-3), (276, 80, 1e-4)]
fine_tuning = [(None, 100, 1e-3)]

imgaug_augmentor = iaa.Sequential([
    iaa.CoarseDropout(0.05)
])

# imgaug_augmentor = get_heavy_augmentor()

# Image augmentations
keras_augmentor = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    # channel_shift_range=50.0,
    # brightness_range=[0.9, 1.1],
    # preprocessing_function=flip_channels,
    # vertical_flip=True,
    # horizontal_flip=True
)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            return # TODO: Seems to only work for FISH_HEAD_DIRECTION
            map_metric(embedding_model, X_train, y_train, X_test, y_test)


# Load static dataset
if USE_DATASET == FISH:
    if DATASET_FLAVOUR == FISH_HEAD_DIRECTION:
        DATASET_DIR = "../data/dataset_condensed/cropped_head/direction"
        with open("../data/classes_condensed_head.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_MALES:
        DATASET_DIR = "../data/dataset_condensed/males/cropped_head/direction"
        with open("../data/classes_condensed_males.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_HEAD_LEFT:
        DATASET_DIR = "../data/dataset_condensed/cropped_head/direction_left"
        with open("../data/classes_condensed_head_left.txt") as file:
            classes = [line.strip() for line in file]

    if DATASET_FLAVOUR == FISH_WHOLE:
        DATASET_DIR = "../data/dataset_condensed/cropped/direction"
        with open("../data/classes_condensed_direction.txt") as file:
            classes = [line.strip() for line in file]

    X_train, y_train = get_images(DATASET_DIR + "\\train", classes, IMG_SIZE)
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test, y_test = get_images(DATASET_DIR + "\\test", classes, IMG_SIZE)
    X_test, y_test = np.array(X_test), np.array(y_test)

if USE_DATASET == OMNIGLOT:  # CHANGE IMG_SIZE to 105
    X, y = get_omniglot()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    classes = np.unique(np.concatenate((y_train, y_test)))

if USE_DATASET == FRUIT:
    DATASET_DIR = "../data/fruit"
    with open("../data/classes_fruit.txt") as file:
        classes = [line.strip() for line in file]

    X, y = get_images(DATASET_DIR, classes, IMG_SIZE)
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# X = np.concatenate((X_train, X_test))
# y = np.concatenate((y_train, y_test))

# fetch dataset generators
if NETWORK_TYPE == SIAMESE_BINARY or NETWORK_TYPE == SIAMESE_CONTRASTIVE:
    train_batch, test_batch = define_siamese_train_test_generators(X_train, y_train, X_test, y_test, BATCH_SIZE,
                                                           keras_augmentor, imgaug_augmentor)
if NETWORK_TYPE == SIAMESE_TRIPLET:
    train_batch, test_batch = define_triplet_train_test_generators(X_train, y_train, X_test, y_test,
                                                                   BATCH_SIZE, keras_augmentor, imgaug_augmentor)
if NETWORK_TYPE == OHNM_TRIPLET:
    train_batch, test_batch = define_ohnm_train_test_generators(X_train, y_train, X_test, y_test,
                                                                BATCH_SIZE, keras_augmentor, imgaug_augmentor)

unique, counts = np.unique(y_train, return_counts=True)
print("Training set size: ", len(y_train))
print("Training set classes with at least two samples: ", (counts >= 2).sum())
unique, counts = np.unique(y_test, return_counts=True)
print("Test set size: ", len(y_test))
print("Test set classes with at least two samples: ", (counts >= 2).sum())

# Show examples of input
samples = next(train_batch)
if not NETWORK_TYPE == SIAMESE_TRIPLET:
    fig, ax = plt.subplots(nrows=4, ncols=2)
    for i, row in enumerate(ax):
        if NETWORK_TYPE == SIAMESE_BINARY or NETWORK_TYPE == SIAMESE_CONTRASTIVE:
            row[0].imshow(samples[0][0][i].reshape(IMG_SHAPE) + 0.5)
        if NETWORK_TYPE == OHNM_TRIPLET:
            row[0].imshow(samples[0][i].reshape(IMG_SHAPE) + 0.5)
        if USE_DATASET == FISH:
            row[1].text(0, 0, str(classes[samples[1][i]]))
    plt.show()
else:
    fig, ax = plt.subplots(nrows=3, ncols=4)
    for i, row in enumerate(ax):
        row[0].imshow(samples[0][0][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
        row[1].imshow(samples[0][1][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
        row[2].imshow(samples[0][2][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
        if USE_DATASET == FISH:
            row[3].text(0, 0.5, str(classes[samples[1][i]]))
    plt.show()

# Select and compile model
if NETWORK_TYPE == SIAMESE_BINARY:
    model = models.siamese_network_binary
    embedding_model = models.embedding_network_binary
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[above_threshold])
if NETWORK_TYPE == SIAMESE_CONTRASTIVE:
    model = models.siamese_network_contrastive
    embedding_model = models.embedding_network_contrastive
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=tfa.losses.contrastive_loss)
if NETWORK_TYPE == SIAMESE_TRIPLET:
    model = models.siamese_network_triplet
    embedding_model = models.embedding_network_triplet
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=triplet_loss)
if NETWORK_TYPE == OHNM_TRIPLET:
    model = models.triplet_network_ohnm
    embedding_model = models.triplet_network_ohnm
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=tfa.losses.TripletSemiHardLoss())

model.summary()

# Calculate epoch sizes
steps_per_epoch = len(X_train) // BATCH_SIZE
test_steps = len(X_test) // BATCH_SIZE

if RETRAIN:

    for sess in fine_tuning:
        unfreeze_point = sess[0]
        n_epochs = sess[1]
        lr = sess[2]

        if unfreeze_point:
            for layer in model.layers[unfreeze_point:]:
                layer.trainable = True

        if NETWORK_TYPE == SIAMESE_BINARY:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss="binary_crossentropy",
                          metrics=[above_threshold])
        if NETWORK_TYPE == SIAMESE_CONTRASTIVE:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=tfa.losses.contrastive_loss)
        if NETWORK_TYPE == SIAMESE_TRIPLET:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=triplet_loss)
        if NETWORK_TYPE == OHNM_TRIPLET:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=tfa.losses.TripletSemiHardLoss())

        model.fit(train_batch, epochs=n_epochs, steps_per_epoch=steps_per_epoch, validation_data=test_batch,
                  validation_steps=test_steps, callbacks=[CustomCallback()])

    model.save_weights('./models/embedding')
else:
    model.load_weights('./models/embedding')

model.evaluate(test_batch, steps=test_steps)

# Show examples of predictions
# samples, labels = next(test_batch)
# prediction = model.predict(samples)
#
# fig, ax = plt.subplots(nrows=3, ncols=3)
# for i, row in enumerate(ax):
#     row[0].imshow(samples[0][i].reshape(IMG_SHAPE)+0.5)
#     row[1].imshow(samples[1][i].reshape(IMG_SHAPE)+0.5)
#     row[2].text(0, 0, "Pred: " + str(prediction[i]) + "(" + str(prediction[i] > 0.5) + ")")
#     row[2].text(0, 0.5, "Label: " + str(labels[i]))
#
# plt.show()
