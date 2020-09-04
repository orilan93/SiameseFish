"""
Script for extracting data from the drive.
"""

import pandas as pd
import os
import shutil
import numpy as np
from collections import defaultdict

SOURCE_DIR = 'E:\\Mugshots'
DEST_DIR = '../data'
DATASET_DIR = os.path.join(DEST_DIR, "dataset")
#FILENAME = "Metadata.txt"
FILENAME = "photoFeltGrønngylt3.csv"
COPY_FILES = True
WRITE_FILES = True
VERBOSITY = 1

# Keep track of number of images for each pit for file naming
name_counter = defaultdict(int)

# Load earlier dataset, if it exists
dataset_file = os.path.join(DEST_DIR, "dataset.txt")
dataset_exists = False
if os.path.isfile(dataset_file):
    dataset_exists = True
    df_dataset = pd.read_csv(dataset_file,names=["filename", "pit", "row_number"])
    df_dataset["count"] = df_dataset.apply(lambda row: row[0].split(".")[0].split("_")[1], axis=1)
    df_dataset["count"] = df_dataset["count"].astype(int)
    df_counter = df_dataset.groupby("pit").max()["count"]
    df_counter = df_counter.apply(lambda x: x + 1)
    name_counter = defaultdict(int, df_counter.to_dict())

dateparser = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')

# Load dataset metadata
df = pd.read_csv(os.path.join(SOURCE_DIR, FILENAME),
                 #encoding='ansi',
                 encoding='utf-8',
                 #sep='\t',
                 sep=';',
                 parse_dates=['date'],
                 date_parser=dateparser)

# Preprocessing of metadata and removal of erroneous data
df.drop(df.tail(2).index,inplace=True)
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
df = df[df['species'] == 'grønngylt']
#df = df[df['sex'] == 'm']
df['images_count'] = df['photostop'] - df['photostart'] + 1
df = df[(df['images_count'] > 1) & (df['images_count'] <= 3)]
df = df[df['pit'].apply(lambda x: x.isnumeric())]
df['pit'] = pd.to_numeric(df['pit'])

# Mapping from period and dataset to folder structure
dataset_map = {
    ("LabMain", "Oct"): "labOct2019",
    ("LabMain", "May"): "labMai2020",
    ("FieldMain", "2"): "Periode2 (P1 2018)",
    ("FieldMain", "2,5"): "Periode2.5 (ekstramerking 2018)",
    ("FieldMain", "3"): "Periode3 (P2 2018)",
    ("FieldMain", "4"): "Periode4 (P3 2018)",
    ("FieldMain", "9"): "Periode9 (P1 2020)"
}

# CSV with image_name, pit
if WRITE_FILES:
    dataset_file = open(os.path.join(DEST_DIR, "dataset.txt"), "a")

error_count = 0
success_count = 0

# Go through all fields in dataset
for index, row in df.iterrows():

    # Skip if record already exists in database
    if dataset_exists:
        if int(row["RowNumber"]) in df_dataset["row_number"].values:
            continue

    try:  # Try a certain dataset and period combination on the dataset mapping
        key = (row['dataset'], row['Period'])
        value = dataset_map[key]
        dataset_path = os.path.join(SOURCE_DIR, value)
    except KeyError:
        if VERBOSITY >= 2:
            print("The key " + str(key) + " does not exist.")
        continue

    # Get all photo id's for each dataset record
    photostart = int(row['photostart'])
    if np.isnan(row['photostop']):
        photostop = photostart
    else:
        photostop = int(row['photostop'])
    photo_ids = list(range(photostart, photostop + 1))

    # Filename constituents
    prefix = "P"
    extension = ".JPG"
    day = str(row['day']).zfill(2)
    month = str(row['month'])
    zfill_num = 4

    # True if row contains full name
    full_name = False

    # Exceptions
    if value == 'labOct2019':
        prefix = "PA"
        month = ""

    # Exceptions
    if value == 'Periode3 (P2 2018)':
        if row['date'] < pd.Timestamp(2018, 7, 6):
            dataset_path = os.path.join(dataset_path, '02.07-05.07')
        else:
            dataset_path = os.path.join(dataset_path, '06.07-09.07')

    # Exceptions
    if value == 'Periode2 (P1 2018)':
        full_name = True
        if row['date'] == pd.Timestamp(2018, 5, 11):
            dataset_path = os.path.join(dataset_path, 'Bilder 11.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 12):
            prefix = "_11"
            month = ""
            day = ""
            zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 12.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 13):
            prefix = "_11"
            month = ""
            day = ""
            zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 13.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 14):
            prefix = "_11"
            month = ""
            day = ""
            zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 14.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 15):
            prefix = "_11"
            month = ""
            day = ""
            zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 15.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 16):
            prefix = "_"

            # Looks like these should work, but are omitted because it's not certain
            prefix = "_11"
            month = ""
            day = ""
            zfill_num = 5

            dataset_path = os.path.join(dataset_path, 'Bilder 16.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 17):
            prefix = "_11"
            month = ""
            day = ""
            zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 17.5.18')

    # Exceptions
    if value == 'Periode2.5 (ekstramerking 2018)':
        full_name = True
        if row['Prefix'] == "P5300":
            dataset_path = os.path.join(dataset_path, '20180530')
        elif row['Prefix'] == "P5310":
            dataset_path = os.path.join(dataset_path, '20180531')
        elif row['Prefix'] == "P6010":
            dataset_path = os.path.join(dataset_path, '20180601')
        elif row['Prefix'] == "P6030":
            dataset_path = os.path.join(dataset_path, '20180603')

    # Special case if full name is provided
    if full_name:
        photostart = row['photostartFull']
        photostop = row['photostopFull']
        prefix = photostart[0]
        photostart = int(photostart[1:])
        photostop = int(photostop[1:])
        photo_ids = [prefix + str(id) for id in range(photostart, photostop + 1)]

    # Go through all photos for a certain database record
    for id in photo_ids:
        id_string = str(id).zfill(zfill_num)
        file_name = prefix + month + day + id_string + extension
        if full_name:
            file_name = id + extension
        file_path = os.path.join(dataset_path, file_name)

        if not os.path.isfile(file_path):
            if VERBOSITY >= 1:
                print("Failed to open " + file_path)
            error_count += 1
        else:  # If file exists

            pit = str(row['pit'])
            count = name_counter[pit]
            out_filename = pit + "_" + str(count) + ".jpg"
            name_counter[pit] += 1

            out_path = os.path.join(DATASET_DIR, out_filename)
            if os.path.isfile(out_path):
                print(out_path, "already exists.")
                continue

            # Create record in new database
            if WRITE_FILES:
                dataset_file.write(out_filename + "," + pit + "," + row['RowNumber'] + "\n")
            if COPY_FILES:
                shutil.copy2(file_path, out_path)

            success_count += 1

# Display the number of successes
print(str(success_count) + "/" + str(error_count + success_count))

if WRITE_FILES:
    dataset_file.close()

    with open(os.path.join(DEST_DIR, "classes.txt"), "w") as classes_file:
        for key in name_counter.keys():
            classes_file.write(str(key) + "\n")
