import models
from data import get_images, define_ohnm_train_test_generators, get_mnist, get_omniglot
from config import IMG_SIZE, NUM_CHANNELS, IMG_SHAPE
import matplotlib.pyplot as plt
import numpy as np
from metrics import map_metric
import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATASET_DIR = "../data/dataset_condensed/cropped_head/direction_left"
RETRAIN = True
BATCH_SIZE = 16
IDENTIFY_HARD_SAMPLES = True

with open("../data/classes_condensed_head_left.txt") as file:
    classes = [line.strip() for line in file]

# X_train, y_train = get_images(DATASET_DIR + "\\train", classes, IMG_SIZE)
# X_train = np.array(X_train)
# y_train = np.array(y_train)
#
# X_test, y_test = get_images(DATASET_DIR + "\\test", classes, IMG_SIZE)
# X_test = np.array(X_test)
# y_test = np.array(y_test)

X, y = get_omniglot()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classes = np.unique(np.concatenate((y_train, y_test)))

X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

# X, y = get_images(DATASET_DIR + "\\train3", classes, IMG_SIZE)
# X = np.array(X)
# y = np.array(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=False)

train_batch, test_batch = define_ohnm_train_test_generators(X_train, y_train, X_test, y_test, BATCH_SIZE)

samples = next(train_batch)

fig, ax = plt.subplots(nrows=4, ncols=2)
for i, row in enumerate(ax):
    row[0].imshow(samples[0][i].reshape(IMG_SHAPE) + 0.5)
    # row[1].text(0, 0, str(classes[samples[1][i]]))

plt.show()


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # closeness_metric(model, X_test, y_test)
        map_metric(model, X_train, y_train, X_test, y_test)


model = models.triplet_network_ohnm
model.summary()
model.compile(tf.keras.optimizers.Adam(1e-3),
              loss=tfa.losses.TripletSemiHardLoss(margin=1.0))  # , distance_metric="angular"))
# Virker som TripletSemiHardLoss ikke skjønner at hvis samme bilde blir valgt igjen så er det samme klasse

steps_per_epoch = len(X_train) // BATCH_SIZE
test_steps = len(X_test) // BATCH_SIZE

# fine_tuning = [(None, 10, 1e-3), (276, 10, 1e-3), (248, 80, 1e-3)]
# fine_tuning_cutoff = [(None, 10), (164, 90)]
fine_tuning = [(None, 10, 1e-3)]

if RETRAIN:

    for sess in fine_tuning:
        unfreeze_point = sess[0]
        n_epochs = sess[1]
        lr = sess[2]

        if unfreeze_point:
            for layer in model.layers[unfreeze_point:]:
                layer.trainable = True

            model.compile(tf.keras.optimizers.Adam(lr),
                          loss=tfa.losses.TripletSemiHardLoss())

        model.fit(train_batch, epochs=n_epochs, steps_per_epoch=steps_per_epoch, validation_data=test_batch,
                  validation_steps=test_steps, callbacks=[CustomCallback()])

    model.save_weights('./models/triplet_ohnm')
else:
    model.load_weights('./models/triplet_ohnm')

model.evaluate(test_batch, steps=test_steps)

if IDENTIFY_HARD_SAMPLES:
    collected = []
    for i in range(100):
        batch = next(test_batch)
        # data = batch[0][:3]
        data = batch[0]
        # label = batch[1][:3]
        label = batch[1]
        embeddings = model.predict(data)
        loss = tfa.losses.triplet_semihard_loss(label, embeddings)
        loss = loss.numpy()
        collected.append((data, label, loss))
    collected.sort(key=lambda x: x[2], reverse=True)

    # fig, ax = plt.subplots(nrows=3, ncols=4)
    # for i, row in enumerate(ax):
    #     row[0].imshow(collected[i][0][0].reshape(IMG_SIZE, IMG_SIZE, NUM_CHANNELS) + 0.5)
    #     row[1].imshow(collected[i][0][1].reshape(IMG_SIZE, IMG_SIZE, NUM_CHANNELS) + 0.5)
    #     row[2].imshow(collected[i][0][2].reshape(IMG_SIZE, IMG_SIZE, NUM_CHANNELS) + 0.5)
    #     row[3].text(0, 0, "Loss: " + str(collected[i][2]))
    #     row[3].text(0, 0.5, str(classes[collected[i][1][0]]) + "\n" + str(classes[collected[i][1][1]]) + "\n" + str(classes[collected[i][1][2]]))

    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle(str(collected[0][2]))
    for i, row in enumerate(ax):
        row[0].imshow(collected[i//2][0][i*4 % 2].reshape(IMG_SHAPE) + 0.5)
        row[1].imshow(collected[i//2][0][(i*4 % 2) + 1].reshape(IMG_SHAPE) + 0.5)
        row[2].imshow(collected[i//2][0][(i*4 % 2) + 3].reshape(IMG_SHAPE) + 0.5)
        row[3].imshow(collected[i//2][0][(i*4 % 2) + 4].reshape(IMG_SHAPE) + 0.5)

    plt.savefig("hard.png")
    plt.show()
    plt.close()

    # fig, ax = plt.subplots(nrows=3, ncols=4)
    # for i, row in enumerate(ax):
    #     row[0].imshow(collected[-(i + 1)][0][0].reshape(IMG_SIZE, IMG_SIZE, NUM_CHANNELS) + 0.5)
    #     row[1].imshow(collected[-(i + 1)][0][1].reshape(IMG_SIZE, IMG_SIZE, NUM_CHANNELS) + 0.5)
    #     row[2].imshow(collected[-(i + 1)][0][2].reshape(IMG_SIZE, IMG_SIZE, NUM_CHANNELS) + 0.5)
    #     row[3].text(0, 0, "Loss: " + str(collected[-(i + 1)][2]))
    #     row[3].text(0, 0.5, str(classes[collected[-(i + 1)][1][0]]) + "\n" + str(classes[collected[-(i + 1)][1][1]]) + "\n" + str(classes[collected[-(i + 1)][1][2]]))

    fig, ax = plt.subplots(nrows=4, ncols=4)
    fig.suptitle(str(collected[-1][2]))
    for i, row in enumerate(ax):
        row[0].imshow(collected[-1 - (i//2)][0][i*4 % 2].reshape(IMG_SHAPE) + 0.5)
        row[1].imshow(collected[-1 - (i//2)][0][(i*4 % 2) + 1].reshape(IMG_SHAPE) + 0.5)
        row[2].imshow(collected[-1 - (i//2)][0][(i*4 % 2) + 3].reshape(IMG_SHAPE) + 0.5)
        row[3].imshow(collected[-1 - (i//2)][0][(i*4 % 2) + 4].reshape(IMG_SHAPE) + 0.5)

    plt.savefig("easy.png")
    plt.show()
    plt.close()
