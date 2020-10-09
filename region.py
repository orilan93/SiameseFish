"""
Network that learns to detect certain regions in an image.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import glob
import random
import os

import models
from config import IMG_SIZE
from models import region_detector
import tensorflow_addons as tfa

from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

DATASET_DIR = os.path.join("..", "data", "newest")
BATCH_SIZE = 32
RETRAIN = True
DEMO = True
PREDICT_BODY = True


def draw_rectangle(image, prediction):
    im = Image.fromarray(np.uint8(image * 255)).convert('RGB')
    # prediction = prediction * 416
    x = prediction[0] * IMG_SIZE
    y = prediction[1] * IMG_SIZE
    w = prediction[2] * IMG_SIZE
    h = prediction[3] * IMG_SIZE
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w
    y2 = y + h
    ImageDraw.Draw(im, "RGBA") \
        .rectangle(((x1, y1), (x2, y2)), fill=None, outline="red", width=10)
    return np.array(im)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Letterboxing
    return tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)


def transform_letterbox(tensor):
    tensor = tf.math.add(tensor, tf.constant([0, (1 - (3 / 4)) / 2, 0, 0]))
    tensor = tf.math.multiply(tensor, tf.constant([1, 3 / 4, 1, 3 / 4]))
    return tensor


def process_path(file_path):
    label = tf.io.read_file(file_path)
    label = tf.strings.split(label, sep="\n")[1]
    label = tf.strings.to_number(tf.strings.split(label))[1:]
    # label = tf.where(tf.equal(tf.string.split(label)[0],"0"))
    # label = tf.gather(label, 1, axis=1, batch_dims=-1)
    # label = tf.math.equal(tf.strings.split(label)[0], tf.constant("0"))
    label = transform_letterbox(label)
    img_path = tf.strings.split(file_path, sep='.txt')[0] + ".jpg"
    img = tf.io.read_file(img_path)
    img = decode_img(img)
    return img, label


def show_batch(image_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.axis('off')
    plt.show()


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


if os.path.isfile(os.path.join(DATASET_DIR, "classes.txt")):
    if os.path.isfile(os.path.join(DATASET_DIR, "classes.bak")):
        os.remove(os.path.join(DATASET_DIR, "classes.bak"))
    os.rename(os.path.join(DATASET_DIR, "classes.txt"), os.path.join(DATASET_DIR, "classes.bak"))

with open("../data/classes.txt") as file:
    classes = [line.strip() for line in file]

list_ds = tf.data.Dataset.list_files(os.path.join(DATASET_DIR, "*.txt"))
image_count = len(list(glob.glob(os.path.join(DATASET_DIR, '*.jpg'))))

model = models.region_detector

if RETRAIN:
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    train_ds = prepare_for_training(labeled_ds)

    #normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    #normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    image_batch, label_batch = next(iter(train_ds))

    if DEMO:
        demo = zip(image_batch.numpy(), label_batch.numpy())
        demo_result = []
        for item in demo:
            image = draw_rectangle(item[0], item[1])
            demo_result.append(image)
        show_batch(demo_result)

    model.compile(optimizer='adam',
                  loss=tfa.losses.giou_loss,
                  #loss=tfa.losses.giou_loss,
                  metrics=['mae', 'mse'])

    model.fit(train_ds, epochs=100, steps_per_epoch=50)
    model.save_weights("models/region")
else:
    model.load_weights("models/region")

# def process_evaluation(file_path):
#    img = tf.io.read_file(file_path)
#    img = decode_img(img)
#    return img


jpg_files = glob.glob(os.path.join(DATASET_DIR, "*.jpg"))
random.shuffle(jpg_files)
jpg_files = jpg_files[:25]

X_test = []
for jpg in jpg_files:
    im = tf.io.read_file(jpg)
    data = decode_img(im)
    X_test.append(data.numpy())
X_test = np.array(X_test)
print(X_test.shape)

# evaluate_ds = tf.data.Dataset.list_files(DATASET_DIR + "/*.jpg")
# evaluate_ds = evaluate_ds.map(process_evaluation, num_parallel_calls=AUTOTUNE)
# evaluate_ds = prepare_for_training(evaluate_ds)

# batch = evaluate_ds.take(25)

prediction = model.predict(X_test)

images = []
for i, im in enumerate(X_test):
    images.append(draw_rectangle(im, prediction[i]))

show_batch(images)
