"""
Visualizes augmented data from the generators for inspection.
"""

import glob
import os
from PIL import Image
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import imgaug.augmenters as iaa

from data import resize_padding, define_simclr_train_test_generators, \
    define_ohnm_train_test_generators
from config import IMG_SIZE
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join("..", "data", "dataset", "cropped_head", "direction")

jpg_files = glob.glob(os.path.join(DATASET_DIR, "*.jpg"))
jpg_files = jpg_files[:16]

with open("../data/classes_direction.txt") as file:
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
    rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #shear_range=0.1,
    zoom_range=0.1,
    # channel_shift_range=50.0,
    # brightness_range=[0.9, 1.1],
    #preprocessing_function=extract_pattern,
    # vertical_flip=True,
    # horizontal_flip=True
)
from imgaug import parameters as iap
imgaug_augmentor = iaa.Sequential([
    #iaa.CoarseDropout(0.05),
    iaa.AddToHueAndSaturation(iap.Uniform(-20, 20)),
    #iaa.CropAndPad(percent=(-0.25, 0.25))
])

gen, _ = define_ohnm_train_test_generators(X, y, X, y, 16, augmentor, imgaug_augmentor)

batch = next(gen)
fig, ax = plt.subplots(nrows=4, ncols=4)
[axi.set_axis_off() for axi in ax.ravel()]
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        col.imshow(batch[0][i*4 + j].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
        col.text(0, 0, str(classes[batch[1][i*4 + j]]))

plt.show()