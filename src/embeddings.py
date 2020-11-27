"""
Network that learns good embeddings from images.
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from config import IMG_SHAPE, IMG_SIZE, BATCH_SIZE
from data import get_omniglot, load_dataset
from metrics import map_metric, above_threshold, triplet_loss, ntxent_loss
from preprocessing.siamese import define_siamese_train_test_generators
from preprocessing.triplet import define_triplet_train_test_generators
from preprocessing.ohnm import define_ohnm_train_test_generators, coverage
from preprocessing.ntxent import define_ntxent_train_test_generators
import models
import const

# Configs
RETRAIN = True
MARGIN = 1.0  # 0.2 # 1.0
GREYSCALE = False
NETWORK_TYPE = const.OHNM_TRIPLET
USE_DATASET = const.FISH
DATASET_FLAVOUR = const.FISH_MERGED
CONTINUE = None  # 'models/model_500' # NONE
CHECKPOINT_AFTER = 100  # None

# Runs through fine tuning steps specified by (unfreeze from layer, epochs, learning rate)
# fine_tuning = [(None, 50, 1e-3), (276, 50, 1e-4), (248, 50, 1e-5)]
# fine_tuning = [(None, 10, 1e-3), (276, 90, 1e-4)]
# fine_tuning = [(None, 10, 1e+3), ([2, 0, 171], 5, 1e+3), ([2, 0, 171], 5, 1e+3)]
# fine_tuning = [(None, 3000, 1e-3)]
# fine_tuning = [(None, 10, 1e-3), (293, 40, 1e-4)]
# fine_tuning = [(None, 1000, 1e-3), (86, 500, 1e-4), (63, 500, 1e-4)]
# fine_tuning = [(86, 100, 1e-4), (63, 100, 1e-4)]
fine_tuning = [(86, 100, 1e-3)]

# Imgaug augmentations
imgaug_augmentor = iaa.Sequential([
    # iaa.CoarseDropout(0.05),
    # iaa.AddToHueAndSaturation(iap.Uniform(-45, 45)),
    iaa.AddToHueAndSaturation(iap.Uniform(-20, 20)),
    # iaa.CropAndPad(percent=(-0.25, 0.25))
    # iaa.Affine(rotate=iap.Uniform(-45, 45))
])

# Keras augmentations
keras_augmentor = ImageDataGenerator(
    rotation_range=10,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # shear_range=0.1,
    zoom_range=0.1,
    # channel_shift_range=50.0,
    # brightness_range=[0.9, 1.1],
    # preprocessing_function=flip_channels,
    # vertical_flip=True,
    # horizontal_flip=True
)

# Load static dataset
if USE_DATASET == const.FISH:
    X_train, y_train, X_test, y_test, classes = load_dataset(DATASET_FLAVOUR)

if USE_DATASET == const.OMNIGLOT:  # CHANGE IMG_SIZE to 105
    X, y = get_omniglot()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    classes = np.unique(np.concatenate((y_train, y_test)))

# fetch dataset generators
if NETWORK_TYPE == const.SIAMESE_BINARY or NETWORK_TYPE == const.SIAMESE_CONTRASTIVE:
    train_batch, test_batch = define_siamese_train_test_generators(X_train, y_train, X_test, y_test, BATCH_SIZE,
                                                                   keras_augmentor, imgaug_augmentor)
if NETWORK_TYPE == const.SIAMESE_NTXENT:
    train_batch, test_batch = define_ntxent_train_test_generators(X_train, y_train, X_test, y_test, BATCH_SIZE,
                                                                  keras_augmentor, imgaug_augmentor)
if NETWORK_TYPE == const.SIAMESE_TRIPLET:
    train_batch, test_batch = define_triplet_train_test_generators(X_train, y_train, X_test, y_test,
                                                                   BATCH_SIZE, keras_augmentor, imgaug_augmentor)
if NETWORK_TYPE == const.OHNM_TRIPLET:
    train_batch, test_batch = define_ohnm_train_test_generators(X_train, y_train, X_test, y_test,
                                                                BATCH_SIZE, keras_augmentor, imgaug_augmentor)

# Prints dataset stats
unique, counts = np.unique(y_train, return_counts=True)
print("Training set size: ", len(y_train))
print("Training set classes with at least two samples: ", (counts >= 2).sum())
unique, counts = np.unique(y_test, return_counts=True)
print("Test set size: ", len(y_test))
print("Test set classes with at least two samples: ", (counts >= 2).sum())

# Show examples of input
samples, labels = next(train_batch)
if NETWORK_TYPE in [const.SIAMESE_BINARY, const.SIAMESE_CONTRASTIVE, const.SIAMESE_NTXENT]:
    fig, ax = plt.subplots(nrows=4, ncols=3)
    [axi.set_axis_off() for axi in ax.ravel()]
    for i, row in enumerate(ax):
        row[0].imshow(samples[0][i].reshape(IMG_SHAPE) + 0.5)
        row[1].imshow(samples[1][i].reshape(IMG_SHAPE) + 0.5)
        if USE_DATASET == const.FISH:
            if NETWORK_TYPE == const.SIAMESE_NTXENT:
                row[2].text(0, 0, str(classes[labels[i]]))
            else:
                row[2].text(0, 0, str(labels[i]))
    plt.show()
if NETWORK_TYPE == const.OHNM_TRIPLET:
    n = math.ceil(math.sqrt(BATCH_SIZE))
    fig, ax = plt.subplots(nrows=n, ncols=n)
    [axi.set_axis_off() for axi in ax.ravel()]
    break_out = False
    for i, row in enumerate(ax):
        if break_out:
            break
        for j, col in enumerate(row):
            ix = i * n + j
            if ix > BATCH_SIZE - 1:
                break_out = True
                break
            col.imshow(samples[ix].reshape(IMG_SHAPE) + 0.5)
            if USE_DATASET == const.FISH:
                col.text(0, 0, str(classes[labels[ix]]))
    plt.show()
if NETWORK_TYPE == const.SIAMESE_TRIPLET:
    fig, ax = plt.subplots(nrows=3, ncols=4)
    [axi.set_axis_off() for axi in ax.ravel()]
    for i, row in enumerate(ax):
        row[0].imshow(samples[0][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
        row[1].imshow(samples[1][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
        row[2].imshow(samples[2][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
        if USE_DATASET == const.FISH:
            row[3].text(0, 0.5, str(classes[labels[i]]))
    plt.show()

# Select and compile model
if NETWORK_TYPE == const.SIAMESE_BINARY:
    model = models.siamese_network_binary
    embedding_model = models.embedding_network_binary
    loss = "binary_crossentropy"
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=loss,
                  metrics=[above_threshold])
if NETWORK_TYPE == const.SIAMESE_CONTRASTIVE:
    model = models.siamese_network_contrastive
    embedding_model = models.embedding_network_contrastive
    loss = tfa.losses.contrastive_loss
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=loss)
if NETWORK_TYPE == const.SIAMESE_NTXENT:
    model = models.siamese_network_ntxent
    embedding_model = models.embedding_network_ntxent
    loss = ntxent_loss
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=loss)
if NETWORK_TYPE == const.SIAMESE_TRIPLET:
    model = models.siamese_network_triplet
    embedding_model = models.embedding_network_triplet
    loss = triplet_loss
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=loss)
if NETWORK_TYPE == const.OHNM_TRIPLET:
    model = models.triplet_network_ohnm
    embedding_model = models.triplet_network_ohnm
    loss = tfa.losses.TripletHardLoss(MARGIN)
    model.compile(tf.keras.optimizers.Adam(1e-3),
                  loss=loss)

if CONTINUE:
    embedding_model.load_weights(CONTINUE)


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if CONTINUE:
            map_metric(embedding_model, X_train, y_train, X_test, y_test, ignore_new=True)

    def on_train_end(self, logs=None):
        map_metric(embedding_model, X_train, y_train, X_test, y_test, ignore_new=True)

    def on_epoch_end(self, epoch, logs=None):

        if CHECKPOINT_AFTER:
            if epoch % CHECKPOINT_AFTER == 0:
                model.save_weights("../models/checkpoint_" + str(epoch))

        if epoch % 5 == 0:
            # return
            map_metric(embedding_model, X_train, y_train, X_test, y_test, ignore_new=True)


model.summary()

# Calculate epoch sizes
steps_per_epoch = len(X_train) // BATCH_SIZE
test_steps = len(X_test) // BATCH_SIZE

if RETRAIN:

    # Execute fine tunings steps
    for sess in fine_tuning:
        unfreeze_point = sess[0]
        n_epochs = sess[1]
        lr = sess[2]

        if unfreeze_point:
            if not type(unfreeze_point) is list:
                unfreeze_point = [unfreeze_point]

            inner_layer = model
            for i in range(len(unfreeze_point) - 1):
                point = unfreeze_point[i]
                inner_layer = inner_layer.layers[point]
            point = unfreeze_point[-1]
            for layer in inner_layer.layers[point:]:
                layer.trainable = True

        if NETWORK_TYPE == const.SIAMESE_BINARY:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=loss,
                          metrics=[above_threshold])
        if NETWORK_TYPE == const.SIAMESE_CONTRASTIVE:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=loss)
        if NETWORK_TYPE == const.SIAMESE_NTXENT:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=loss)
        if NETWORK_TYPE == const.SIAMESE_TRIPLET:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=loss)
        if NETWORK_TYPE == const.OHNM_TRIPLET:
            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=loss)

        # TODO: NTXENT trenger mest sansynlig ikke validation
        model.fit(train_batch, epochs=n_epochs, steps_per_epoch=steps_per_epoch, validation_data=test_batch,
                  validation_steps=test_steps, callbacks=[CustomCallback()])

    # Save model after training
    model.save_weights('../models/contrastive')
    if DATASET_FLAVOUR == const.FISH_PAIR_LEFT:
        embedding_model.save_weights('../models/left')
    elif DATASET_FLAVOUR == const.FISH_PAIR_RIGHT:
        embedding_model.save_weights('../models/right')
    else:
        embedding_model.save_weights('../models/embedding')
else:
    model.load_weights('../models/contrastive')

model.evaluate(test_batch, steps=test_steps)

# Show examples of predictions
samples, labels = next(test_batch)
prediction = model.predict(samples)
if NETWORK_TYPE in [const.SIAMESE_BINARY, const.SIAMESE_CONTRASTIVE]:
    fig, ax = plt.subplots(nrows=3, ncols=3)
    [axi.set_axis_off() for axi in ax.ravel()]
    for i, row in enumerate(ax):
        row[0].imshow(samples[0][i].reshape(IMG_SHAPE) + 0.5)
        row[1].imshow(samples[1][i].reshape(IMG_SHAPE) + 0.5)
        if NETWORK_TYPE == const.SIAMESE_BINARY:
            row[2].text(0, 0, "Pred: " + str(prediction[i][0]) + "(" + str(prediction[i][0] > 0.5) + ")")
            row[2].text(0, 0.5, "Label: " + str(labels[i]))
        if NETWORK_TYPE in [const.SIAMESE_CONTRASTIVE]:
            row[2].text(0, 0, "Pred: " + str(prediction[i]))
            row[2].text(0, 0.5, "Label: " + str(labels[i]))
    plt.show()
if NETWORK_TYPE == const.SIAMESE_TRIPLET:
    fig, ax = plt.subplots(nrows=3, ncols=4)
    [axi.set_axis_off() for axi in ax.ravel()]
    for i, row in enumerate(ax):
        row[0].imshow(samples[0][i].reshape(IMG_SHAPE) + 0.5)
        row[1].imshow(samples[1][i].reshape(IMG_SHAPE) + 0.5)
        row[2].imshow(samples[2][i].reshape(IMG_SHAPE) + 0.5)
        pred = np.split(prediction[i], 3)
        dist_pos = np.linalg.norm(np.subtract(pred[0], pred[1]))
        dist_neg = np.linalg.norm(np.subtract(pred[0], pred[2]))
        dist = np.linalg.norm(dist_pos - dist_neg)
        row[3].text(0, 0, "Dist: " + str(dist))
        row[3].text(0, 0.5, "Label: " + str(labels[i]))
    plt.show()
if NETWORK_TYPE == const.OHNM_TRIPLET:
    collected = []
    for i in range(100):
        batch = samples, labels
        # data = batch[0][:3]
        data = batch[0]
        # label = batch[1][:3]
        label = batch[1]
        embeddings = model.predict(data)
        loss_val = loss(label, embeddings).numpy()
        collected.append((data, label, loss_val))
    collected.sort(key=lambda x: x[2], reverse=True)

    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle(str(collected[0][2]))
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.imshow(collected[0][0][i * 4 + j].reshape(IMG_SHAPE) + 0.5)
            col.text(0, 0, classes[collected[0][1][i * 4 + j]])

    plt.show()
    plt.close()

    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle(str(collected[-1][2]))
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.imshow(collected[-1][0][i * 4 + j].reshape(IMG_SHAPE) + 0.5)
            col.text(0, 0, classes[collected[-1][1][i * 4 + j]])

    plt.show()
    plt.close()

    # n = min(math.ceil(math.sqrt(BATCH_SIZE)), 4)
    # fig, ax = plt.subplots(nrows=n, ncols=n)
    # [axi.set_axis_off() for axi in ax.ravel()]
    # loss = loss(labels, prediction).numpy()
    # fig.suptitle("loss: " + str(loss))
    # break_out = False
    # for i, row in enumerate(ax):
    #     if break_out:
    #         break
    #     for j, col in enumerate(row):
    #         ix = i*n+j
    #         if ix > BATCH_SIZE - 1:
    #             break_out = True
    #             break
    #         col.imshow(samples[ix].reshape(IMG_SHAPE) + 0.5)
    # plt.show()

print("Dataset coverage: ", len(coverage) / len(classes))
