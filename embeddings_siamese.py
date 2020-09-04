import datetime
import tensorflow as tf
import models
from data import get_images, define_siamese_train_test_generators, get_omniglot
from config import IMG_SIZE, IMG_SHAPE
import matplotlib.pyplot as plt
import numpy as np
from metrics import above_threshold
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

# DATASET_DIR = "D:\Jobb\Siamese Fish\data\dataset_condensed\cropped_head\direction"
DATASET_DIR = "../data/dataset_condensed/cropped_head/direction"
RETRAIN = True
BATCH_SIZE = 8

# with open("../data/classes_condensed_head.txt") as file:
with open("../data/classes_condensed_head.txt") as file:
    classes = [line.strip() for line in file]

#classes = list(set([os.path.splitext(os.path.basename(jpg_file))[0].split("_")[0] for jpg_file in glob.glob(DATASET_DIR + "\\*.jpg")]))

#with open('../data/classes_condensed2.txt', 'w') as f:
#    for item in classes:
#        f.write("%s\n" % item)

# X, y = get_images(DATASET_DIR, classes, IMG_SIZE)

X_train, y_train = get_images(DATASET_DIR + "\\train", classes, IMG_SIZE)
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test, y_test = get_images(DATASET_DIR + "\\test", classes, IMG_SIZE)
X_test = np.array(X_test)
y_test = np.array(y_test)

# X, y = get_omniglot()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# classes = np.unique(np.concatenate((y_train, y_test)))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=0)

# X_test_pairs, y_test_pairs = gen_pairs_n(X_test, y_test, n=50, seed=1)
# X_test_pairs = [np.array(list(x)) for x in list(zip(*X_test_pairs))]
# X_test_pairs, y_test_pairs = shuffle(X_test_pairs, y_test_pairs, random_state=0)
# X_test_pairs = [np.array(list(x)) for x in list(zip(*X_test_pairs))]

# train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=False)

train_batch, test_batch = define_siamese_train_test_generators(X_train, y_train, X_test, y_test, BATCH_SIZE)

# (imgs, labels), dummy = next(train_aug)
samples = next(train_batch)

fig, ax = plt.subplots(nrows=3, ncols=3)
for i, row in enumerate(ax):
    row[0].imshow(samples[0][0][i].reshape(IMG_SHAPE) + 0.5)
    row[1].imshow(samples[0][1][i].reshape(IMG_SHAPE) + 0.5)
    row[2].text(0, 0, str(samples[1][i]))

plt.show()

model = models.siamese_network_contrastive
embedding_model = models.embedding_network_contrastive
model.summary()
model.compile(optimizer='adam',
              #loss=tfa.losses.ContrastiveLoss(margin=1.0),
              loss='binary_crossentropy',
              metrics=[above_threshold])

log_dir = "logs\\"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

steps_per_epoch = len(X_train) // BATCH_SIZE
test_steps = len(X_test) // BATCH_SIZE

if RETRAIN:
    # X_train_pairs, y_train_pairs = get_pairs(X, y, n=100)
    # dataset = tf.data.Dataset.from_generator(
    #     lambda: gen_pairs(X_train, y_train),
    #     (tf.float32, tf.bool),
    #     (tf.TensorShape([None, IMG_SIZE, IMG_SIZE, 3]), tf.TensorShape([])))

    # dataset_generator = gen_pairs(X_train, y_train)

    # X_train_pairs, y_train_pairs = gen_pairs_n(X_train, y_test, 250)#200)
    # X_train_pairs = [np.array(list(x)) for x in list(zip(*X_train_pairs))]
    # X_train_pairs, y_train_pairs = shuffle(X_train_pairs, y_train_pairs, random_state=0)
    # X_train_pairs = [np.array(list(x)) for x in list(zip(*X_train_pairs))]
    # X_train_pairs, y_train_pairs = get_pairs(X, y, 1000)
    # model.fit(X_train_pairs, y_train_pairs, epochs=20, batch_size=8, validation_data=(X_test_pairs, y_test_pairs))#,
    #          callbacks=[tensorboard_callback])

    model.fit(train_batch, epochs=10, steps_per_epoch=steps_per_epoch, validation_data=test_batch,
              validation_steps=test_steps)  # , callbacks=[tensorboard_callback])

    # history = model.fit_generator(train_aug, steps_per_epoch=steps_per_epoch,
    #                              epochs=epochs, verbose=1,
    #                              validation_data=valid_aug, validation_steps=validation_steps,
    #                              shuffle=True)
    # model.fit(train_x, train_y, epochs=10, validation_data=(valid_x, valid_y))
    model.save_weights('./models/siamese')
    embedding_model.save_weights('./models/embedding')
else:
    model.load_weights('./models/siamese')
    embedding_model.load_weights('./models/embedding')

model.evaluate(test_batch, steps=test_steps)

# one_epoch = list(itertools.islice(train_batch, steps_per_epoch))
# predictions = model.predict(one_epoch[0])
# from sklearn.metrics import accuracy_score
# print(accuracy_score(one_epoch[1], predictions))

# test = gen_pairs_n(X_test, y_test, 3, seed=3)
# x_test = test[0]
# prediction = model.predict(x_test)
#
# fig, ax = plt.subplots(nrows=3, ncols=3)
# for i, row in enumerate(ax):
#     row[0].imshow(x_test[0][i].reshape(IMG_SIZE, IMG_SIZE, 3))
#     row[1].imshow(x_test[1][i].reshape(IMG_SIZE, IMG_SIZE, 3))
#     row[2].text(0, 0, str(prediction[i]))
#
# plt.show()

samples, labels = next(test_batch)
prediction = model.predict(samples)

fig, ax = plt.subplots(nrows=3, ncols=3)
for i, row in enumerate(ax):
    row[0].imshow(samples[0][i].reshape(IMG_SHAPE)+0.5)
    row[1].imshow(samples[1][i].reshape(IMG_SHAPE)+0.5)
    row[2].text(0, 0, "Pred: " + str(prediction[i]) + "(" + str(prediction[i] < 0.5) + ")")
    row[2].text(0, 0.5, "Label: " + str(labels[i]))

plt.show()
