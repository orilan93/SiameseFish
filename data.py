"""
Functions for retrieving and processing data.
"""

import random
import numpy as np
import itertools
from PIL import Image
import glob
import os
from itertools import combinations
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
import tensorflow_datasets as tdfs
from config import IMG_SIZE
from collections import defaultdict
import imgaug.augmenters as iaa
import imgaug as ia


def get_images(path, classes, image_size):
    """Gets the images in the specified folder in the form of X and y."""
    X = []
    y = []
    jpg_files = glob.glob(path + "\\*.jpg")
    for jpg_file in jpg_files:
        basename = os.path.basename(jpg_file)
        basename_wo_ext = os.path.splitext(basename)[0]
        basename_split = basename_wo_ext.split(sep='_')
        label = "_".join(basename_split[:-1])
        #label = basename_split[0]
        index = classes.index(label)

        pil_img = Image.open(jpg_file)
        pil_img = resize_padding(pil_img, image_size)

        X.append(np.array(pil_img))
        y.append(index)
    return X, y


def get_mnist(n_samples=100):
    """Gets a subset of the mnist dataset."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train[:n_samples]
    X_test = X_test[:n_samples]
    y_train = y_train[:n_samples]
    y_test = y_test[:n_samples]

    X_train_new = []
    X_test_new = []

    for i, x in enumerate(X_train):
        im = Image.fromarray(x)
        im = resize_padding(im, IMG_SIZE)
        X_train_new.append(np.array(im))

    for i, x in enumerate(X_test):
        im = Image.fromarray(x)
        im = resize_padding(im, IMG_SIZE)
        X_test_new.append(np.array(im))

    X_train = np.array(X_train_new)
    X_test = np.array(X_test_new)

    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    X_train = np.tile(X_train, (1, 1, 3))
    X_test = np.tile(X_test, (1, 1, 3))
    return X_train, X_test, y_train, y_test


def get_omniglot(name="small1"):
    """Gets the omniglot dataset, but does not resize the images."""
    omniglot = tdfs.load(name="omniglot")
    gen = tdfs.as_numpy(omniglot)
    data = []
    for d in gen['small1']:
        data.append(d)

    X = []
    y = []
    for d in data:
        X.append(d['image'])
        y.append(d['label'])
    X = np.array(X)
    y = np.array(y)
    return X, y


def get_pairs(x, y, n=10, seed=0):
    """Returns pairs from the dataset and whether the pair is similar or not."""
    random.seed(seed)
    x_1 = []
    x_2 = []
    y_1 = []
    length = len(y)
    for i in range(n):
        i1 = random.randrange(length)
        i2 = random.randrange(length)
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[i1]
        y2 = y[i2]
        not_same = (y1 != y2)
        x_1.append(x1)
        x_2.append(x2)
        y_1.append(not_same)
    return [np.array(x_1), np.array(x_2)], np.array(y_1)


def get_equal_pairs(y):
    equal_pairs = []
    uniques = set(y)
    for unique in uniques:
        same_indices = [i for i, x in enumerate(y) if x == unique]
        pairs = list(combinations(same_indices, 2))
        equal_pairs += pairs
    return equal_pairs


def gen_pairs(x, y, seed=0):
    """Generator for generating pairs from the dataset and whether the pair is similar or not."""
    random.seed(seed)
    length = len(y)

    equal_pairs = get_equal_pairs(y)
    random.shuffle(equal_pairs)
    equal_pair_i = 0

    while True:
        # equal pair
        if equal_pair_i < len(equal_pairs):
            i1 = equal_pairs[equal_pair_i][0]
            i2 = equal_pairs[equal_pair_i][1]
            x1 = x[i1]
            y1 = y[i1]
            x2 = x[i2]
            y2 = y[i2]
            equal_pair_i += 1
            #if equal_pair_i >= len(equal_pairs) - 1:
                #equal_pair_i = 0
                #random.shuffle(equal_pairs)

            yield [np.array(x1), np.array(x2)], y1 == y2

        # random pair
        i1 = random.randrange(length)
        i2 = random.randrange(length)
        while i2 == i1:
            i2 = random.randrange(length)
        x1 = x[i1]
        x2 = x[i2]
        y1 = y[i1]
        y2 = y[i2]
        yield [np.array(x1), np.array(x2)], y1 == y2


def gen_pairs_n(X, y, n, seed=0):
    """Generates n pairs from the dataset and whether the pair is similar or not."""
    dataset = list(itertools.islice(gen_pairs(X, y, seed=seed), n))
    X_pairs, y_pairs = zip(*dataset)
    X_pairs = list(X_pairs)
    X_pairs = [np.array(list(x)) for x in list(zip(*X_pairs))]
    y_pairs = np.array(list(y_pairs))
    return X_pairs, y_pairs


def get_embeddings2(model, samples, targets):
    """Returns embeddings from images."""

    # Group together samples with same target
    grouped = defaultdict(list)
    for i in range(len(samples)):
        grouped[targets[i]].append(samples[i])

    # Make a dictionary of lists of embeddings with target as key
    results = defaultdict()
    for k in grouped.keys():
        result = model.predict(np.array(grouped[k]))
        results[k] = result

    # Flatten the dictionary values as X
    X = np.concatenate(list(results.values()))
    # Restore y from dictionary keys
    y = [[k] * len(results[k]) for k in results.keys()]
    # Flatten the y list
    y = [item for sublist in y for item in sublist]

    return X, y


def get_embeddings(model, samples, targets):
    """Returns embeddings from images."""

    X = model.predict(samples)

    y = targets

    return X, y


def resize_padding(img, size):
    """Resizes a PIL image to target size with padding (letterboxing)."""
    old_size = img.size
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    pil_img = img.resize(new_size, Image.ANTIALIAS)
    new_img = Image.new(img.mode, (size, size))
    new_img.paste(pil_img, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return new_img


def get_heavy_augmentor():
    """Gets a imgaug augmentor with quite heavy augmentations."""

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.

    return iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Invert(0.05, per_channel=True),  # invert color channels
                           iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-4, 0),
                                   first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                   second=iaa.LinearContrast((0.5, 2.0))
                               )
                           ]),
                           iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           iaa.Grayscale(alpha=(0.0, 1.0)),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )


def to_greyscale(x):
    """Transforms image to greyscale"""
    return np.expand_dims(np.dot(x[..., :3], [0.2989, 0.5870, 0.1140]), axis=2)


def extract_pattern(x):
    """Extracts pattern as binary image."""
    x = x.astype(np.uint8)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(x)
    saliencyMap_u8 = (saliencyMap * 255).astype('uint8')
    ret, threshMap = cv2.threshold(saliencyMap_u8, 32, 255, cv2.THRESH_BINARY)
    # threshMap = cv2.threshold(saliencyMap_u8, 0, 255,
    #                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    threshMap = np.expand_dims(threshMap, axis=2)
    return threshMap


def saliency_map(x):
    """Converts the image into a saliency map."""
    x = x.astype(np.uint8)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(x)
    saliencyMap_u8 = (saliencyMap * 255).astype(np.uint8)
    saliencyMap_u8 = np.expand_dims(saliencyMap_u8, axis=2)
    return saliencyMap_u8


def flip_channels(x):
    """Randomly flips colors channels,"""
    x = np.einsum("ijk->kij", x)
    np.random.shuffle(x)
    x = np.einsum("kij->ijk", x)
    return x


#####################################
# Siamese data processing functions #
#####################################


def input_generator_siamese(x, y):
    data_len = len(x)
    k = 0
    last_idx = -1
    last_y = -1

    while True:

        idx = np.random.choice(range(data_len))

        if k == 3:# or k == 5 or k==7:
            same_id = [j for j, e in enumerate(y) if e == last_y and j != last_idx]
            if not same_id:
                idx = np.random.choice(range(data_len))
            else:
                idx = np.random.choice(same_id)


        #if k == 7:
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

        similarity_labels = list(map(lambda pair: pair[0] == pair[1], labels_pair))

        yield data_pair, np.array(similarity_labels)


def define_siamese_train_test_generators(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()  # rescale=1.0 / 255.0)
    train_input_gen = input_generator_siamese(train_x, train_y)
    test_input_gen = input_generator_siamese(test_x, test_y)

    train_aug = augmentation_generator_siamese(train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_aug = augmentation_generator_siamese(test_input_gen, test_data_generator, batch_size)

    train_batch = pair_generator_siamese(train_aug)
    test_batch = pair_generator_siamese(test_aug)

    return train_batch, test_batch


#####################################
# Triplet data processing functions #
#####################################


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


##################################
# OHNM data processing functions #
##################################


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


def augmentation_generator_ohnm(input_gen, data_gen, batch_size=16, imgaug_augmentor=None):
    while True:
        batch = list(itertools.islice(input_gen, batch_size))
        data, labels = zip(*batch)
        data = np.array(list(data))
        labels = np.array(list(labels))

        unique, counts = np.unique(labels, return_counts=True)
        if np.any(counts > 2):
            continue
        if not np.any(counts > 1):
            continue

        if imgaug_augmentor:
            data = imgaug_augmentor(images=data)
        x = data_gen.flow(data, batch_size=batch_size, shuffle=False)
        x = next(x)
        x = tf.keras.applications.inception_v3.preprocess_input(x)
        #x = tf.keras.applications.vgg16.preprocess_input(x)

        yield x, labels


def define_ohnm_train_test_generators(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()#preprocessing_function=to_greyscale)  # rescale=1.0 / 255.0)
    train_input_gen = input_generator_ohnm(train_x, train_y)
    test_input_gen = input_generator_ohnm(test_x, test_y)

    train_batch = augmentation_generator_ohnm(train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_batch = augmentation_generator_ohnm(test_input_gen, test_data_generator, batch_size)

    return train_batch, test_batch


def define_ohnm_train_test_generators2(train_x, train_y, test_x, test_y, batch_size, keras_augmentor=ImageDataGenerator(), imgaug_augmentor=None):
    train_data_generator = keras_augmentor
    test_data_generator = ImageDataGenerator()#preprocessing_function=to_greyscale)  # rescale=1.0 / 255.0)
    train_input_gen = input_generator_siamese2(train_x, train_y)
    test_input_gen = input_generator_siamese2(test_x, test_y)

    train_batch = augmentation_generator_siamese(train_input_gen, train_data_generator, batch_size, imgaug_augmentor)
    test_batch = augmentation_generator_siamese(test_input_gen, test_data_generator, batch_size)

    return train_batch, test_batch