"""
Tests different aspects of the system.
"""

import glob
import os

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


#test_pair_generation()
#test_lonely_annotations()

print("All tests finished.")