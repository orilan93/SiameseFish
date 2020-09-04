"""
Script that preprocesses the dataset such that it is ready for training.
Generally each part of the program preprocesses the input and puts it into subdirectories.

COPY_FILES: Filters files that is deemed useful and stores them in a directory.
CROP_FILES: Crops images based on meta files.
LABEL_DIRECTION: Names the files based on their directions.
CREATE_CLASS_FILE: Creates a file that contains all classes based on the filenames in the input directory.
TEST_SPLIT: Splits a dataset into a test and train set.
EXTRACT_MALES: Consults the dataset metafile to identify and extract all male individuals in a given directory.
"""

import os
import glob
from shutil import copy2
from PIL import Image
from collections import defaultdict
import random
import pandas as pd

# Where to get the data and where to put it
DATA_DIR = '../data'
SRC_DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DST_DATASET_DIR = os.path.join(DATA_DIR, "dataset_condensed")
CROPPED_DATASET_DIR = os.path.join(DST_DATASET_DIR, "cropped")
DIRECTION_DATASET_DIR = os.path.join(CROPPED_DATASET_DIR, "direction")
TRAIN_DATASET_DIR = os.path.join(DIRECTION_DATASET_DIR, "train")
TEST_DATASET_DIR = os.path.join(DIRECTION_DATASET_DIR, "test")
MALE_DIRECTORY = os.path.join(DST_DATASET_DIR, "males")

# The types of preprocessing to be done
COPY_FILES = False
CROP_FILES = False
LABEL_DIRECTION = False
CREATE_CLASS_FILE = False
TEST_SPLIT = True
EXTRACT_MALES = False

# Various other parameters
OVERWRITE = False
TEST_SPLIT_FRACTION = 0.3
TENSORFLOW_FORMAT = False # TODO: Implement this
IGNORE_DIRECTION = None  # 'l', 'r', None
CROP_BODY = False
META_FILE = 'E:\\Mugshots\\photoFeltGrønngylt2.csv'
ENSURE_PAIRS = False
RANDOM_SWAP = False

if COPY_FILES:
    target_individuals = glob.glob(SRC_DATASET_DIR + "/*_3.jpg")

    for individual in target_individuals:
        pit = os.path.basename(individual)
        pit = os.path.splitext(pit)[0]
        pit = pit.split(sep='_')[0]
        individual_images = glob.glob(SRC_DATASET_DIR + "/" + pit + "_*.jpg")
        for img_file in individual_images:
            dst_path = os.path.join(DST_DATASET_DIR, os.path.basename(img_file))
            copy2(img_file, dst_path)
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            if os.path.isfile(txt_file):
                dst_path = os.path.join(DST_DATASET_DIR, os.path.basename(txt_file))
                copy2(txt_file, dst_path)

if CROP_FILES:
    txt_files = glob.glob(DST_DATASET_DIR + "/[!classes]*.txt")
    for txt_file in txt_files:
        img_file = os.path.splitext(txt_file)[0] + ".jpg"
        basename = os.path.basename(img_file)
        out_path = os.path.join(CROPPED_DATASET_DIR, basename)

        if not OVERWRITE and not os.path.isfile(out_path):
            with open(txt_file) as file:
                line = file.readlines()[int(CROP_BODY)].split()
                x = float(line[1]) * 4000
                y = float(line[2]) * 3000
                w = float(line[3]) * 4000
                h = float(line[4]) * 3000
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x1 + w)
                y2 = int(y1 + h)

            img = Image.open(img_file)
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img.save(out_path)

if LABEL_DIRECTION:
    filename_counter = defaultdict(int)
    with open(os.path.join(DATA_DIR, "direction_condensed.txt")) as file:
        for line in file:
            line = line.strip().split(sep=',')
            filename = line[0]
            direction = "l" if line[1] == "0" else "r"
            if direction == IGNORE_DIRECTION:
                continue
            pit = filename.split(sep='_')[0]
            out_filename = pit + "_" + direction
            filename_counter[out_filename] += 1
            out_filename += "_" + str(filename_counter[out_filename]) + ".jpg"
            src_path = os.path.join(CROPPED_DATASET_DIR, filename)
            if not os.path.isfile(src_path):
                print("Warning:", src_path, "does not exist")
                continue
            dst_path = os.path.join(DIRECTION_DATASET_DIR, out_filename)
            if not OVERWRITE and not os.path.isfile(dst_path):
                copy2(src_path, dst_path)
    with open(os.path.join(DATA_DIR, "classes_condensed_males_direction.txt"), "w") as file:
        for key in filename_counter.keys():
            file.write(key + "\n")

if CREATE_CLASS_FILE:
    jpg_files = glob.glob(DIRECTION_DATASET_DIR + "/*.jpg")
    classes = list(set(["_".join(os.path.basename(jpg_file).split("_")[:2]) for jpg_file in jpg_files]))
    with open(os.path.join(DATA_DIR, "classes_condensed_males_direction.txt"), "w") as file:
        for c in classes:
            file.write(c + "\n")


def random_swap(l1, l2, n):
    l1_len = len(l1)
    l2_len = len(l2)

    for i in range(n):
        idx1 = random.randrange(l1_len)
        idx2 = random.randrange(l2_len)

        a = l1[idx1]
        l1[idx1] = l2[idx2]
        l2[idx2] = a


if TEST_SPLIT:
    jpg_files = glob.glob(DIRECTION_DATASET_DIR + "/*.jpg")

    if ENSURE_PAIRS:
        classes = list(set(["_".join(os.path.basename(jpg_file).split("_")[:2]) for jpg_file in jpg_files]))
        random.shuffle(classes)
        test_n = int(len(classes)*TEST_SPLIT_FRACTION)
        train_classes = classes[test_n:]
        test_classes = classes[:test_n]
        train = [jpg_file for jpg_file in jpg_files if "_".join(os.path.basename(jpg_file).split("_")[:2]) in train_classes]
        test = [jpg_file for jpg_file in jpg_files if "_".join(os.path.basename(jpg_file).split("_")[:2]) in test_classes]
        if RANDOM_SWAP:
            random_swap(train, test, 100)
    else:
        random.shuffle(jpg_files)
        test_n = int(len(jpg_files)*TEST_SPLIT_FRACTION)
        train = jpg_files[test_n:]
        test = jpg_files[:test_n]

    for t in train:
        basename = os.path.basename(t)
        dst_path = os.path.join(TRAIN_DATASET_DIR, basename)
        if not OVERWRITE and not os.path.isfile(dst_path):
            copy2(t, dst_path)
    for t in test:
        basename = os.path.basename(t)
        dst_path = os.path.join(TEST_DATASET_DIR, basename)
        if not OVERWRITE and not os.path.isfile(dst_path):
            copy2(t, dst_path)

if EXTRACT_MALES:

    df = pd.read_csv(META_FILE, sep=';')
    jpg_files = glob.glob(DST_DATASET_DIR + "\\*.jpg")
    pits = [os.path.splitext(os.path.basename(jpg_file))[0].split('_')[0] for jpg_file in jpg_files]

    df = df[df['sex'] == 'm']
    df = df[df['pit'].isin(pits)]

    male_pits = df['pit'].tolist()

    copy_files = []

    for pit in male_pits:
        g = DST_DATASET_DIR + "\\" + pit + "*"
        files = glob.glob(g)
        copy_files.extend(files)

    for file in copy_files:
        basename = os.path.basename(file)
        dst_path = os.path.join(MALE_DIRECTORY, basename)
        copy2(file, dst_path)