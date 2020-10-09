import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data import get_images, get_embeddings
from config import IMG_SIZE, DESCRIPTOR_SIZE
import models
import pandas as pd

# DATASET_DIR = "../data/dataset/cropped_head/direction"
#
# BATCH_SIZE = 5
# RETRAIN = False
# DEMO = False
# SAVE_EMBEDDINGS = False
# # PREDICT_BODY = True
#
# with open("../data/classes_direction.txt") as file:
#     classes = [line.strip() for line in file]

# jpg_files = glob.glob(DATASET_DIR + "/*.jpg")
# classes = list(set(["_".join(os.path.basename(jpg_file).split("_")[:2]) for jpg_file in jpg_files]))

# list_ds = tf.data.Dataset.list_files(DATASET_DIR + "\\*.jpg")
# for file in list_ds.take(1):
#     print("File: ", file.numpy())
#
#
# def decode_img(img):
#     # convert the compressed string to a 3D uint8 tensor
#     img = tf.image.decode_jpeg(img, channels=3)
#     # Use `convert_image_dtype` to convert to floats in the [0,1] range.
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     # Letterboxing
#     return tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
#
#
# def process_path(file_path):
#     class_name = tf.strings.split(file_path, sep=os.path.sep, name="get_basename")[-1]
#     class_name = tf.strings.split(class_name, sep=".", name="remove_extension")[0]
#     class_name = tf.strings.split(class_name, sep="_", name="split_seperator")[:2]
#     class_name = tf.unstack(class_name, num=2)
#     class_name = tf.strings.join(class_name, '_')
#     label = class_name == classes
#     label = tf.where(label)
#     label = label[0]
#     img = tf.io.read_file(file_path)
#     img = decode_img(img)
#     return img, label
#
#
# def prepare_for_training(ds, cache=True, shuffle_buffer_size=1):
#     # This is a small dataset, only load it once, and keep it in memory.
#     # use `.cache(filename)` to cache preprocessing work for datasets that don't
#     # fit in memory.
#     if cache:
#         if isinstance(cache, str):
#             ds = ds.cache(cache)
#         else:
#             ds = ds.cache()
#
#     ds = ds.shuffle(buffer_size=shuffle_buffer_size)
#
#     # Repeat forever
#     ds = ds.repeat()
#
#     ds = ds.batch(BATCH_SIZE)
#
#     # `prefetch` lets the dataset fetch batches in the background while the model
#     # is training.
#     ds = ds.prefetch(buffer_size=AUTOTUNE)
#
#     return ds
#
#
# labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#
# for image, label in labeled_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())
#
# train_ds = prepare_for_training(labeled_ds)

# X, y = get_images(DATASET_DIR, classes, IMG_SIZE)

# X_train, y_train = get_images(DATASET_DIR + "\\train", classes, IMG_SIZE)
# X_train = np.array(X_train)
# X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
# y_train = np.array(y_train)
#
# X_test, y_test = get_images(DATASET_DIR + "\\test", classes, IMG_SIZE)
# X_test = np.array(X_test)
# X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)
# y_test = np.array(y_test)


# X = np.array(X)
# #X = X.astype('float32')
# #X /= 255.0
# #X = tf.keras.applications.vgg16.preprocess_input(X)
# X = tf.keras.applications.inception_v3.preprocess_input(X)
# y = np.array(y)

# X_train, y_train = get_images(os.path.join(DATASET_DIR, "train"), classes, IMG_SIZE)
# X_train, y_train = np.array(X_train), np.array(y_train)
# X_train = tf.keras.applications.inception_v3.preprocess_input(X_train)
#
# X_test, y_test = get_images(os.path.join(DATASET_DIR, "test"), classes, IMG_SIZE)
# X_test, y_test = np.array(X_test), np.array(y_test)
# X_test = tf.keras.applications.inception_v3.preprocess_input(X_test)
#
# if SAVE_EMBEDDINGS:
#     embedding_model = models.embedding_network_contrastive
#     embedding_model.load_weights('./models/embedding')
#     X_embeddings, y_embeddings = get_embeddings(embedding_model, X, y)
#     columns = ["x" + str(i) for i in range(DESCRIPTOR_SIZE)] + ["y"]
#     df = pd.DataFrame(np.column_stack([X_embeddings, y_embeddings]), columns=columns)
#     df = df.astype({"y": int})
#     df.to_csv("embeddings/embeddings.csv", index=False)
#
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)
#
# # Set y to -1 if record does not exist in the training set
# for i, y_test_i in enumerate(y_test):
#     if y_test_i not in y_train:
#         y_test[i] = -1
#
# X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)
#
# unique, counts = np.unique(y_valid, return_counts=True)
# print("y_valid new observations: ", str(dict(zip(unique, counts))[-1]))
#
# classifier = EmbeddingSVCClassifier()
# classifier.train(X_train, y_train)
# accuracy = classifier.evaluate(X_test, y_test)
#
# # best_accuracy = 0
# # best_t = -1
# # #for t in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.7, 1.0, 2.0, 5.0, 10.0]:
# # #for t in [0.1, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]:
# # #for t in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
# # for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.1]:
# #     print("t = ", t)
# #     classifier = EmbeddingDistanceClassifier(t)
# #     #classifier.train(X_train[0], y_train[0])
# #     classifier.train(X_train, y_train)
# #     accuracy = classifier.evaluate(X_valid, y_valid)
# #     if accuracy > best_accuracy:
# #         best_accuracy = accuracy
# #         best_t = t
# #     print("======================================")
# #
# # print("best t: ", best_t)
# #
# # classifier = EmbeddingDistanceClassifier(best_t)
# # classifier.train(X_train, y_train)
# # accuracy = classifier.evaluate(X_test, y_test)
#
# best_accuracy = 0.0
# best_k = -1
# for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
#     print("k = ", k)
#     classifier = EmbeddingKNNClassifier(k)
#     classifier.train(X_train, y_train)
#     accuracy = classifier.evaluate(X_valid, y_valid)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_k = k
#     print("======================================")
#
# print("best t: ", best_k)
#
# classifier = EmbeddingKNNClassifier(best_k)
# classifier.train(X_train, y_train)
# accuracy = classifier.evaluate(X_test, y_test)
#
# # classifier = EmbeddingDistanceClassifier(0.7)
# # classifier.train(X_train, y_train)
# # accuracy = classifier.evaluate(X_test, y_test)
#
# prediction = classifier.predict(X_test[:4])
#
# fig, ax = plt.subplots(nrows=4, ncols=2)
# for i, row in enumerate(ax):
#    row[0].imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE, 3) + 0.5)
#    pred_class = -1 if prediction[i] == -1 else classes[prediction[i]]
#    row[1].text(0, 0, "Predicted: " + str(pred_class))
#    row[1].text(0, 0.5, "Ground truth: " + str(y_test[i])) # Check if y is -1
#
# plt.show()
#

