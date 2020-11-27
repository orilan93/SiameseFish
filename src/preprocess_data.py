"""
Script that preprocesses the dataset such that it is ready for training.
Generally each part of the program preprocesses the input and puts it into subdirectories.
Note: It is recommended to understand the code before running this script as it will do file operations.

COPY_FILES: Filters files that is deemed useful and stores them in a directory.
CROP_FILES: Crops images based on meta files.
LABEL_DIRECTION: Names the files based on their directions.
CREATE_LEFT_RIGHT_SPLIT: Splits a dataset into train and test for both left and right while keeping the pairs synced.
CREATE_CLASS_FILE: Creates a file that contains all classes based on the filenames in the input directory.
TEST_SPLIT: Splits a dataset into a test and train set.
EXTRACT_MALES: Consults the dataset metafile to identify and extract all male individuals in a given directory.
EXTRACT_NEWEST: Extracts the samples after a given date.
"""

import os
import glob
from shutil import copy2
from PIL import Image
from collections import defaultdict
import random
import pandas as pd
from datetime import datetime
import numpy as np

DATA_DIR = os.path.join("..", "data")
SRC_DIR = os.path.join(DATA_DIR, "merged", "Oct")
DST_DIR = os.path.join(DATA_DIR, "merged", "Oct")

# The types of preprocessing to be done
COPY_FILES = False
CROP_FILES = False
LABEL_DIRECTION = False
CREATE_LEFT_RIGHT_SPLIT = False
CREATE_CLASS_FILE = False
TEST_SPLIT = True
EXTRACT_MALES = False
EXTRACT_NEWEST = False

# Various other parameters
OVERWRITE = False
TEST_SPLIT_FRACTION = 0.3
TENSORFLOW_FORMAT = False # TODO: Implement this
IGNORE_DIRECTION = 'r'  # 'l', 'r', None
CROP_BODY = False
META_FILE = 'E:\\Mugshots\\photoFeltGrønngylt2.csv'
ENSURE_PAIRS = False
RANDOM_SWAP = False
EXTRACT_AFTER = datetime(2020, 8, 1)
UPDATE_DATASET_DIRECTION = False
DATASET_FILE = os.path.join("..", "data", "dataset.csv")

if COPY_FILES:
    target_individuals = glob.glob(SRC_DIR + "/*_3.jpg")

    for individual in target_individuals:
        pit = os.path.basename(individual)
        pit = os.path.splitext(pit)[0]
        pit = pit.split(sep='_')[0]
        individual_images = glob.glob(SRC_DIR + "/" + pit + "_*.jpg")
        for img_file in individual_images:
            dst_path = os.path.join(DST_DIR, os.path.basename(img_file))
            copy2(img_file, dst_path)
            txt_file = os.path.splitext(img_file)[0] + ".txt"
            if os.path.isfile(txt_file):
                dst_path = os.path.join(DST_DIR, os.path.basename(txt_file))
                copy2(txt_file, dst_path)

if CROP_FILES:
    txt_files = glob.glob(SRC_DIR + "/[!classes]*.txt")
    for txt_file in txt_files:
        img_file = os.path.splitext(txt_file)[0] + ".jpg"
        basename = os.path.basename(img_file)
        out_path = os.path.join(DST_DIR, basename)

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

    if UPDATE_DATASET_DIRECTION:
        df_dataset = pd.read_csv(DATASET_FILE, encoding="latin-1")
        df_dataset["dir_filename"] = ""

    filename_counter = defaultdict(int)
    with open(os.path.join(DATA_DIR, "direction.txt")) as file:
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
            src_path = os.path.join(SRC_DIR, filename)
            if not os.path.isfile(src_path):
                print("Warning:", src_path, "does not exist")
                continue
            dst_path = os.path.join(DST_DIR, out_filename)

            if UPDATE_DATASET_DIRECTION:
                idx = list(np.where(df_dataset["new_filename"] == filename)[0])[0]
                df_dataset["dir_filename"][idx] = out_filename

            if not OVERWRITE and not os.path.isfile(dst_path):
                copy2(src_path, dst_path)
    with open(os.path.join(DATA_DIR, "classes_direction_left.txt"), "w") as file:
        for key in filename_counter.keys():
            file.write(key + "\n")
    if UPDATE_DATASET_DIRECTION:
        df_dataset.to_csv(DATASET_FILE, index=False)


if CREATE_LEFT_RIGHT_SPLIT:
    jpg_files = glob.glob(SRC_DIR + "/*.jpg")
    jpg_files = [os.path.basename(jpg_file) for jpg_file in jpg_files]
    left_files = [jpg_file for jpg_file in jpg_files if jpg_file.split('_')[1] == 'l']
    pairs = []
    for left_file in left_files:
        right_file = left_file.split('_')
        right_file[1] = 'r'
        right_file = '_'.join(right_file)
        if os.path.isfile(os.path.join(DST_DIR, right_file)):
            pairs.append([left_file, right_file])
    random.shuffle(pairs)
    test_n = int(len(pairs) * TEST_SPLIT_FRACTION)
    train = pairs[test_n:]
    test = pairs[:test_n]

    with open(os.path.join(DATA_DIR, "classes_pairs.txt"), "w") as file:
        for pair in pairs:
            cls = pair[0].split('_')[0]
            file.write(cls + "\n")

    for t in train:
        left_file = t[0]
        dst_path = os.path.join(DST_DIR, "direction_left", "train", left_file)
        if not OVERWRITE and not os.path.isfile(dst_path):
            left_file = os.path.join(SRC_DIR, t[0])
            copy2(left_file, dst_path)

        right_file = t[1]
        dst_path = os.path.join(DST_DIR, "direction_right", "train", right_file)
        if not OVERWRITE and not os.path.isfile(dst_path):
            right_file = os.path.join(SRC_DIR, t[1])
            copy2(right_file, dst_path)

    for t in test:
        left_file = t[0]
        dst_path = os.path.join(DST_DIR, "direction_left", "test", left_file)
        if not OVERWRITE and not os.path.isfile(dst_path):
            left_file = os.path.join(SRC_DIR, t[0])
            copy2(left_file, dst_path)

        right_file = t[1]
        dst_path = os.path.join(DST_DIR, "direction_right", "test", right_file)
        if not OVERWRITE and not os.path.isfile(dst_path):
            right_file = os.path.join(SRC_DIR, t[1])
            copy2(right_file, dst_path)


if CREATE_CLASS_FILE:
    jpg_files = glob.glob(SRC_DIR + "/*.jpg")
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
    jpg_files = glob.glob(SRC_DIR + "\\*.jpg")

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
        dst_path = os.path.join(DST_DIR, "train", basename)
        if not OVERWRITE and not os.path.isfile(dst_path):
            copy2(t, dst_path)
    for t in test:
        basename = os.path.basename(t)
        dst_path = os.path.join(DST_DIR, "test", basename)
        if not OVERWRITE and not os.path.isfile(dst_path):
            copy2(t, dst_path)

if EXTRACT_MALES:

    df = pd.read_csv(META_FILE, sep=';')
    jpg_files = glob.glob(SRC_DIR + "\\*.jpg")
    pits = [os.path.splitext(os.path.basename(jpg_file))[0].split('_')[0] for jpg_file in jpg_files]

    df = df[df['sex'] == 'm']
    df = df[df['pit'].isin(pits)]

    male_pits = df['pit'].tolist()

    copy_files = []

    for pit in male_pits:
        g = SRC_DIR + "\\" + pit + "*"
        files = glob.glob(g)
        copy_files.extend(files)

    for file in copy_files:
        basename = os.path.basename(file)
        dst_path = os.path.join(DST_DIR, basename)
        copy2(file, dst_path)

if EXTRACT_NEWEST:

    jpg_files = glob.glob(SRC_DIR + "\\*.jpg")

    copy_files = []

    for jpg_file in jpg_files:
        time = os.path.getmtime(jpg_file)
        if time > EXTRACT_AFTER.timestamp():
            txt_file = os.path.splitext(jpg_file)[0] + ".txt"
            copy_files.append(txt_file)
            copy_files.append(jpg_file)

    for file in copy_files:
        basename = os.path.basename(file)
        dst_path = os.path.join(DST_DIR, basename)
        copy2(file, dst_path)