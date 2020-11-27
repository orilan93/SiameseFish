import itertools
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from preprocessing.ohnm import input_generator_ohnm, augmentation_generator_ohnm


def input_generator_simclr(x, y):
    data_len = len(x)

    while True:

        idx = np.random.choice(range(data_len))
        yield x[idx], y[idx]


def augmentation_generator_simclr(input_gen, data_gen, batch_size=16, imgaug_augmentor=None):
    while True:
        batch = list(itertools.islice(input_gen, batch_size))
        data, labels = zip(*batch)
        data1 = np.array(list(data))
        labels = np.array(list(labels))

        #data2 = copy.deepcopy(data1)
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


def define_simclr_train_test_generators(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()#preprocessing_function=to_greyscale)  # rescale=1.0 / 255.0)
    train_input_gen = input_generator_simclr(train_x, train_y)
    test_input_gen = input_generator_ohnm(test_x, test_y)

    train_batch = augmentation_generator_simclr(train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_batch = augmentation_generator_ohnm(test_input_gen, test_data_generator, batch_size*2)

    return train_batch, test_batch