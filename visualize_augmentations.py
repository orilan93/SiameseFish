"""
Visualizes augmented data from the generators for inspection.
"""

import glob
import os
from PIL import Image
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

from data import resize_padding, define_ohnm_train_test_generators, saliency_map, extract_pattern
from config import IMG_SIZE
import matplotlib.pyplot as plt

DATASET_DIR = "../data/dataset_condensed/cropped_head/direction"

jpg_files = glob.glob(DATASET_DIR + "\\*.jpg")
jpg_files = jpg_files[:16]

with open("../data/classes_condensed_head.txt") as file:
    classes = [line.strip() for line in file]

X = []
y = []
for jpg_file in jpg_files:
    basename = os.path.basename(jpg_file)
    basename_wo_ext = os.path.splitext(basename)[0]
    basename_split = basename_wo_ext.split(sep='_')
    label = "_".join(basename_split[:2])
    #label = basename_split[0]
    index = classes.index(label)

    pil_img = Image.open(jpg_file)
    pil_img = resize_padding(pil_img, IMG_SIZE)

    X.append(np.array(pil_img))
    y.append(index)


augmentor = ImageDataGenerator(
    #rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #shear_range=0.1,
    #zoom_range=0.1,
    # channel_shift_range=50.0,
    # brightness_range=[0.9, 1.1],
    preprocessing_function=extract_pattern,
    # vertical_flip=True,
    # horizontal_flip=True
)

gen, _ = define_ohnm_train_test_generators(X, y, X, y, 16, augmentor)

batch = next(gen)
fig, ax = plt.subplots(nrows=4, ncols=4)
for i, row in enumerate(ax):
    row[0].imshow(batch[0][i*4].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[1].imshow(batch[0][i*4+1].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[2].imshow(batch[0][i*4+2].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
    row[3].imshow(batch[0][i*4+3].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)

plt.show()