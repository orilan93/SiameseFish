"""
Tests different aspects of the system.
"""

import glob
import os
from data import get_images, gen_pairs_n
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATASET_DIR = '../data/dataset_new/cropped_head/direction_left'
DATASET_DIR = '../data/dataset_new'


def test_lonely_annotations():
    txt_files = glob.glob(DATASET_DIR + '/*.txt')
    for txt_file in txt_files:
        jpg_file = os.path.splitext(txt_file)[0] + ".jpg"
        if not os.path.isfile(jpg_file):
            print(jpg_file, "does not exist.")
            #os.remove(txt_file)


def test_annotations():
    txt_files = glob.glob(DATASET_DIR + '/*.txt')
    for txt_file in txt_files:
        with open(txt_file) as file:
            line = file.readline()
            if line[0] != "0":
                print(txt_file, "head is not first")
            line = file.readline()
            if line[0] != "1":
                print(txt_file, "body is not second")


def test_get_conv_layers():
    from utils import get_conv_layers
    from models import embedding_network_contrastive
    conv_layers = get_conv_layers(embedding_network_contrastive)
    print(len(conv_layers))


#test_get_conv_layers()
#test_pair_generation()
#test_lonely_annotations()

print("All tests finished.")