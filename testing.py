"""
Tests different aspects of the system.
"""

import glob
import os
from data import get_images, gen_pairs_n
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATASET_DIR = '../data/dataset'
DATASET_DIR = '../data/dataset_condensed/cropped_head/direction_left'


def test_lonely_annotations():
    txt_files = glob.glob(DATASET_DIR + '/*.txt')
    for txt_file in txt_files:
        jpg_file = os.path.splitext(txt_file)[0] + ".jpg"
        if not os.path.isfile(jpg_file):
            print(jpg_file, "does not exist.")
            #os.remove(txt_file)


def test_annotations_start_with_head():
    txt_files = glob.glob(DATASET_DIR + '/*.txt')
    for txt_file in txt_files:
        with open(txt_file) as file:
            first_line = file.readline()
            if first_line[0] != "0":
                print(txt_file, "does not start with head.")


def test_pair_generation():
    with open("../data/classes_condensed_head_left.txt") as file:
        classes = [line.strip() for line in file]

    X, y = get_images(DATASET_DIR, classes, 416)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=0)
    X_test_pairs, y_test_pairs = gen_pairs_n(X_test, y_test, 3)
    #X_test_pairs, y_test_pairs = gen_pairs_n(X, y, n=3, seed=3)
    #print(classes[62])
    #print(classes[32])
    #print(len(y_test_pairs))
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for i, row in enumerate(ax):
        row[0].imshow(X_test_pairs[0][i].reshape(416, 416, 3))
        row[1].imshow(X_test_pairs[1][i].reshape(416, 416, 3))
        row[2].text(0, 0, "Lol")

    plt.show()


def test_get_conv_layers():
    from utils import get_conv_layers
    from models import embedding_network_contrastive
    conv_layers = get_conv_layers(embedding_network_contrastive)
    print(len(conv_layers))


test_get_conv_layers()
#test_pair_generation()
#test_lonely_annotations()
#test_annotations_start_with_head()

print("All tests finished.")