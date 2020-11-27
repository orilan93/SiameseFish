import itertools
from collections import defaultdict
import tensorflow as tf
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from preprocessing.siamese import input_generator_siamese2, augmentation_generator_siamese


def input_generator_ohnm(x, y):
    data_len = len(x)

    while True:

        idx = np.random.choice(range(data_len))
        yield x[idx], y[idx]

        same_id = [j for j, e in enumerate(y) if e == y[idx] and j != idx]
        if not same_id:
            # continue
            idx = np.random.choice(range(data_len))
        else:
            idx = np.random.choice(same_id)

        yield x[idx], y[idx]


coverage = defaultdict(int)


def augmentation_generator_ohnm(input_gen, data_gen, batch_size=16, imgaug_augmentor=None):
    while True:
        batch = list(itertools.islice(input_gen, batch_size))
        data, labels = zip(*batch)
        data = np.array(list(data))
        labels = np.array(list(labels))

        unique, counts = np.unique(labels, return_counts=True)
        #if np.any(counts > 2):
        #    continue
        if not np.any(counts > 1):
            continue
        #if np.count_nonzero(counts > 1) < len(counts)//2:
        #    continue

        for l in labels:
            coverage[l] = coverage[l] + 1

        if imgaug_augmentor:
            data = imgaug_augmentor(images=data)
        x = data_gen.flow(data, batch_size=batch_size, shuffle=False)
        x = next(x)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        #x = tf.keras.applications.vgg16.preprocess_input(x)

        yield x, labels


def define_ohnm_train_test_generators(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()
    train_input_gen = input_generator_ohnm(train_x, train_y)
    test_input_gen = input_generator_ohnm(test_x, test_y)

    train_batch = augmentation_generator_ohnm(train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_batch = augmentation_generator_ohnm(test_input_gen, test_data_generator, batch_size)

    return train_batch, test_batch


def define_ohnm_train_test_generators2(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()
    train_input_gen = input_generator_siamese2(train_x, train_y)
    test_input_gen = input_generator_siamese2(test_x, test_y)

    train_batch = augmentation_generator_siamese(train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_batch = augmentation_generator_siamese(test_input_gen, test_data_generator, batch_size)

    return train_batch, test_batch