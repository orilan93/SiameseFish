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
import tensorflow as tf
import tensorflow_datasets as tdfs

import const
from config import IMG_SIZE
from collections import defaultdict


def load_images(path, classes, image_size, greyscale=False, direction_labels=True, format="jpg", include_filenames=False):
    """Gets the images in the specified path in the form of X and y.

    The images needs to be on the format of <pit>[_<direction>]_<increment>.<format>
    
    Args:
        path: A directory path containing the images or a list of file paths.
        classes (list): List of classes that y will represent.
        image_size (int): Scale the images to this quadratic size.
        greyscale (bool): Whether to load the image in greyscale.
        direction_labels (bool): Whether the filenames have direction information or not.
        format (string): The image file extension.
        include_filenames (bool): Whether to return the filenames.

    Returns:
        A tuple containing an array with the images and an array containing the corresponding classes. If include_filenames is True, then the filenames are also returned as part of the tuple.
    """
    X = []
    y = []
    filenames = []

    if os.path.isfile(path):
        jpg_files = [path]
    else:
        glob_string = os.path.join(path, "*." + format)
        print(glob_string)
        jpg_files = glob.glob(glob_string)

    for jpg_file in jpg_files:
        basename = os.path.basename(jpg_file)
        basename_wo_ext = os.path.splitext(basename)[0]
        basename_split = basename_wo_ext.split(sep='_')
        if direction_labels:
            label = "_".join(basename_split[:-1])
        else:
            label = basename_split[0]
        index = classes.index(label) #TODO: Sjekke at dette er helt riktig (er det problem hvis det er hull her?)

        pil_img = Image.open(jpg_file)
        if greyscale:
            pil_img = pil_img.convert('L')
        pil_img = resize_padding(pil_img, image_size)
        pil_img = np.array(pil_img)
        if greyscale:
            pil_img = np.expand_dims(pil_img, axis=2)
            pil_img = np.tile(pil_img, (1, 1, 3))

        X.append(pil_img)
        y.append(index)
        filenames.append(basename)
    if include_filenames:
        return X, y, filenames
    else:
        return X, y
    
    
def load_dataset(dataset_flavour, preprocess=False, greyscale=False, include_filenames=False):
    """Loads the dataset given a dataset flavour.

        Args:
            dataset_flavour: The dataset flavour, from types.py.
            preprocess (bool): Whether to run images through inception preprocessing.
            greyscale (bool): Whether to load images as greyscale.
            include_filenames (bool): Whether to return the filenames.

        Returns:
            Return training and test dataset, class names and optionally filenames.
        """

    direction_labels = True
    if dataset_flavour in [const.FISH_MERGED_PAIR_LEFT, const.FISH_MERGED_PAIR_RIGHT, const.FISH_MERGED_LEFT, const.FISH_MERGED_RIGHT, const.FISH_HEAD_PAIR_LEFT, const.FISH_HEAD_PAIR_RIGHT]:
        direction_labels = False

    format = "jpg"
    if dataset_flavour == const.FISH_BINARY:
        format = "png"

    if dataset_flavour == const.FISH_HEAD:
        DATASET_DIR = os.path.join('..', '..', 'data', 'dataset', 'cropped_head', 'direction')
        with open("../../data/classes_direction.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_BINARY:
        DATASET_DIR = os.path.join('..', '..', 'data', 'binary')
        with open("../../data/classes_direction.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_HEAD_LEFT:
        DATASET_DIR = os.path.join('..', '..', 'data', 'dataset', 'cropped_head', 'direction_left')
        with open("../../data/classes_direction_left.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_HEAD_RIGHT:
        DATASET_DIR = os.path.join('..', '..', 'data', 'dataset', 'cropped_head', 'direction_right')
        with open("../../data/classes_direction_right.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_MERGED_PAIR_LEFT:
        DATASET_DIR = os.path.join('..', '..', 'data', 'merged', 'head', 'Oct', 'left')
        with open("../../data/classes.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_MERGED_PAIR_RIGHT:
        DATASET_DIR = os.path.join('..', '..', 'data', 'merged', 'head', 'Oct', 'right')
        with open("../../data/classes.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_HEAD_PAIR_LEFT:
        DATASET_DIR = os.path.join('..', '..', 'data', 'dataset', 'cropped_head', 'direction_left')
        with open("../../data/classes.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_HEAD_PAIR_RIGHT:
        DATASET_DIR = os.path.join('..', '..', 'data', 'dataset', 'cropped_head', 'direction_right')
        with open("../../data/classes.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_WHOLE:
        DATASET_DIR = os.path.join('..', '..', 'data', 'dataset', 'cropped_body', 'direction')
        with open("../../data/classes_direction.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_MERGED:
        DATASET_DIR = os.path.join('..', '..', 'data', 'merged', 'head', 'Oct')
        with open("../../data/classes_direction.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_MERGED_LEFT:
        DATASET_DIR = os.path.join('..', '..', 'data', 'merged', 'head', 'Oct', 'left')
        with open("../../data/classes.txt") as file:
            classes = [line.strip() for line in file]

    if dataset_flavour == const.FISH_MERGED_RIGHT:
        DATASET_DIR = os.path.join('..', '..', 'data', 'merged', 'head', 'Oct', 'right')
        with open("../../data/classes.txt") as file:
            classes = [line.strip() for line in file]

    tup = load_images(os.path.join(DATASET_DIR, "train"), classes, IMG_SIZE, greyscale=greyscale,
                                   direction_labels=direction_labels, format=format, include_filenames=include_filenames)
    X_train, y_train, *rest = tup
    if include_filenames:
        train_filenames = rest[0]
    X_train, y_train = np.array(X_train), np.array(y_train)

    tup = load_images(os.path.join(DATASET_DIR, "test"), classes, IMG_SIZE, greyscale=greyscale,
                                 direction_labels=direction_labels, format=format, include_filenames=include_filenames)
    X_test, y_test, *rest = tup
    if include_filenames:
        test_filenames = rest[0]
    X_test, y_test = np.array(X_test), np.array(y_test)

    if preprocess:
        X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
        X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)

    if include_filenames:
        return X_train, y_train, X_test, y_test, classes, train_filenames, test_filenames
    else:
        return X_train, y_train, X_test, y_test, classes


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
    """Gets the omniglot dataset (does not resize the images)."""

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


def to_greyscale(x):
    """Transforms image to greyscale"""
    return np.expand_dims(np.dot(x[..., :3], [0.2989, 0.5870, 0.1140]), axis=2)


def flip_channels(x):
    """Randomly flips colors channels,"""
    x = np.einsum("ijk->kij", x)
    np.random.shuffle(x)
    x = np.einsum("kij->ijk", x)
    return x
