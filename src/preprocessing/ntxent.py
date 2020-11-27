import itertools
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator


def input_generator_ntxent(x, y):
    data_len = len(x)

    while True:

        idx = np.random.choice(range(data_len))
        yield x[idx], y[idx]


def augmentation_generator_ntxent(input_gen, data_gen, batch_size=16, imgaug_augmentor=None):
    while True:
        batch = list(itertools.islice(input_gen, batch_size))
        data, labels = zip(*batch)
        data1 = np.array(list(data))
        labels = np.array(list(labels))

        unique, counts = np.unique(labels, return_counts=True)
        if np.any(counts > 1):
            # NTXENT can't have two same class in same batch
            continue

        data2 = data1.copy()

        if imgaug_augmentor:
            data1 = imgaug_augmentor(images=data1)
            data2 = imgaug_augmentor(images=data2)

        x = np.concatenate([data1, data2])
        labels = np.tile(labels, 2)

        x = data_gen.flow(x, batch_size=batch_size*2, shuffle=False)
        x = next(x)
        x = tf.keras.applications.inception_v3.preprocess_input(x)

        yield x, labels


def pair_generator_ntxent(aug_gen, batch_size):
    while True:
        data, labels = next(aug_gen)

        data_pair = zip(data[::1], data[batch_size::1])
        data_pair = list(zip(*data_pair))
        data_pair = map(list, data_pair)
        data_pair = list(map(np.array, data_pair))
        labels = labels[:batch_size]
        #labels_pair = list(zip(labels[::1], labels[batch_size::1]))

        #similarity_labels = list(map(lambda pair: pair[0] == pair[1], labels_pair))

        yield data_pair, np.array(labels)


def define_ntxent_train_test_generators(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()#preprocessing_function=to_greyscale)  # rescale=1.0 / 255.0)
    train_input_gen = input_generator_ntxent(train_x, train_y)
    test_input_gen = input_generator_ntxent(test_x, test_y)

    train_aug = augmentation_generator_ntxent(train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_aug = augmentation_generator_ntxent(test_input_gen, test_data_generator, batch_size)

    train_batch = pair_generator_ntxent(train_aug, batch_size)
    test_batch = pair_generator_ntxent(test_aug, batch_size)

    return train_batch, test_batch