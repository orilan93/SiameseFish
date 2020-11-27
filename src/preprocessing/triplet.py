import itertools
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from config import IMG_SIZE
import tensorflow as tf


def input_generator_triplet(x, y):
    data_len = len(x)

    while True:

        anchor_idx = np.random.choice(range(data_len))
        anchor_class = y[anchor_idx]

        pos_indices = [j for j, e in enumerate(y) if e == anchor_class and j != anchor_idx]
        if not pos_indices:
            continue
        pos_idx = np.random.choice(pos_indices)

        neg_indices = [j for j, e in enumerate(y) if e != anchor_class]
        neg_idx = np.random.choice(neg_indices)

        yield [x[anchor_idx], x[pos_idx], x[neg_idx]], anchor_class


def augmentation_generator_triplet(input_gen, data_gen, batch_size=8, imgaug_augmentor=None):
    num_images = batch_size*3
    while True:
        batch = list(itertools.islice(input_gen, batch_size))
        data, labels = zip(*batch)
        data = np.array(list(data))
        data = data.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
        labels = np.array(list(labels))

        if imgaug_augmentor:
            data = imgaug_augmentor(images=data)
        x = data_gen.flow(data, batch_size=num_images, shuffle=False)
        x = next(x)
        x = tf.keras.applications.inception_v3.preprocess_input(x)

        x = x.reshape((-1, 3, IMG_SIZE, IMG_SIZE, 3))
        x = np.einsum("bixyz->ibxyz", x)
        x = [inp for inp in x]

        yield x, labels


def define_triplet_train_test_generators(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()
    train_input_gen = input_generator_triplet(train_x, train_y)
    test_input_gen = input_generator_triplet(test_x, test_y)

    train_batch = augmentation_generator_triplet(train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_batch = augmentation_generator_triplet(test_input_gen, test_data_generator, batch_size)

    return train_batch, test_batch