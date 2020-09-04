import models
from data import get_images, IMG_SIZE, define_triplet_train_test_generators
import matplotlib.pyplot as plt
import numpy as np
from metrics import triplet_loss

DATASET_DIR = "../data/dataset_condensed/cropped_head/direction"
RETRAIN = True
BATCH_SIZE = 8

with open("../data/classes_condensed_head.txt") as file:
    classes = [line.strip() for line in file]

X_train, y_train = get_images(DATASET_DIR + "\\train", classes, IMG_SIZE)
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test, y_test = get_images(DATASET_DIR + "\\test", classes, IMG_SIZE)
X_test = np.array(X_test)
y_test = np.array(y_test)

train_batch, test_batch = define_triplet_train_test_generators(X_train, y_train, X_test, y_test, BATCH_SIZE)

samples = next(train_batch)

fig, ax = plt.subplots(nrows=3, ncols=4)
[axi.set_axis_off() for axi in ax.ravel()]
for i, row in enumerate(ax):
    row[0].imshow(samples[0][0][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[1].imshow(samples[0][1][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[2].imshow(samples[0][2][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[3].text(0, 0.5, str(classes[samples[1][i]]))

plt.show()


model = models.siamese_network_triplet
embedding_model = models.embedding_network_triplet
model.summary()
model.compile(optimizer='adam',
              loss=triplet_loss)

steps_per_epoch = len(X_train) // BATCH_SIZE
test_steps = len(X_test) // BATCH_SIZE

if RETRAIN:

    model.fit(train_batch, epochs=10, steps_per_epoch=steps_per_epoch, validation_data=test_batch,
              validation_steps=test_steps)

    model.save_weights('./models/siamese_triplet')
    embedding_model.save_weights('./models/embedding_triplet')
else:
    model.load_weights('./models/siamese_triplet')
    embedding_model.load_weights('./models/embedding_triplet')

model.evaluate(test_batch, steps=test_steps)

samples, labels = next(test_batch)
prediction = model.predict(samples)

fig, ax = plt.subplots(nrows=3, ncols=4)
[axi.set_axis_off() for axi in ax.ravel()]
for i, row in enumerate(ax):
    row[0].imshow(samples[0][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[1].imshow(samples[1][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[2].imshow(samples[1][i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[3].text(0, 0.5, "Label: " + str(labels[i]))

plt.show()
