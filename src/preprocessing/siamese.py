import itertools
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf


def input_generator_siamese(x, y):
    data_len = len(x)
    k = 0
    last_idx = -1
    last_y = -1

    while True:

        idx = np.random.choice(range(data_len))

        if k == 3:  # or k == 5 or k==7:
            same_id = [j for j, e in enumerate(
                y) if e == last_y and j != last_idx]
            if not same_id:
                idx = np.random.choice(range(data_len))
            else:
                idx = np.random.choice(same_id)

        # if k == 7:
        if k == 3:
            k = -1

        last_y = y[idx]
        last_idx = idx
        k += 1

        yield x[idx], y[idx]


def input_generator_siamese2(x, y):
    data_len = len(x)

    while True:

        while True:
            idx = np.random.choice(range(data_len))
            same_id = [j for j, e in enumerate(y) if e == y[idx] and j != idx]
            if same_id:
                break

        yield x[idx], y[idx]

        idx = np.random.choice(same_id)

        yield x[idx], y[idx]

        idx = np.random.choice(range(data_len))

        yield x[idx], y[idx]

        idx = np.random.choice(range(data_len))

        yield x[idx], y[idx]


def augmentation_generator_siamese(input_gen, data_gen, batch_size=8, imgaug_augmentor=None):
    batch_size = batch_size * 2
    while True:
        batch = list(itertools.islice(input_gen, batch_size))
        data, labels = zip(*batch)
        data = np.array(list(data))
        labels = np.array(list(labels))

        if imgaug_augmentor:
            data = imgaug_augmentor(images=data)
        x = data_gen.flow(data, batch_size=batch_size, shuffle=False)
        x = next(x)
        x = tf.keras.applications.inception_v3.preprocess_input(x)

        yield x, labels


def pair_generator_siamese(aug_gen):
    while True:
        data, labels = next(aug_gen)

        data_pair = zip(data[::2], data[1::2])
        data_pair = list(zip(*data_pair))
        data_pair = map(list, data_pair)
        data_pair = list(map(np.array, data_pair))
        labels_pair = list(zip(labels[::2], labels[1::2]))

        similarity_labels = list(
            map(lambda pair: pair[0] == pair[1], labels_pair))

        yield data_pair, np.array(similarity_labels)


def define_siamese_train_test_generators(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()  # rescale=1.0 / 255.0)
    train_input_gen = input_generator_siamese2(train_x, train_y)
    test_input_gen = input_generator_siamese2(test_x, test_y)

    train_aug = augmentation_generator_siamese(
        train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_aug = augmentation_generator_siamese(
        test_input_gen, test_data_generator, batch_size)

    train_batch = pair_generator_siamese(train_aug)
    test_batch = pair_generator_siamese(test_aug)

    return train_batch, test_batch
