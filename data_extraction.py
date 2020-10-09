"""
Script for extracting data from the drive.
"""

import pandas as pd
import os
import shutil
import numpy as np
from collections import defaultdict

SOURCE_DIR = 'G:\\Mugshots'
DEST_DIR = os.path.join('..', 'data')
DATASET_DIR = os.path.join(DEST_DIR, "dataset_new")
#FILENAME = "Metadata.txt"
FILENAME1 = "photoFeltGrønngylt_complete2020_addbrygga.csv"
FILENAME2 = 'RecapPIerFlødevigen.csv'
DATASET_FILE = os.path.join(DEST_DIR, "dataset_new.txt")
CLASSES_FILE = "classes_new.txt"
COPY_FILES = False
WRITE_FILES = True
SAVE_ERRORS = False
VERBOSITY = 2

# Keep track of number of images for each pit for file naming
name_counter = defaultdict(int)

# Load earlier dataset, if it exists
dataset_exists = False
if os.path.isfile(DATASET_FILE):
    dataset_exists = True
    df_dataset = pd.read_csv(DATASET_FILE, names=["filename", "pit", "løpernummer"])
    df_dataset["count"] = df_dataset.apply(lambda row: row[0].split(".")[0].split("_")[1], axis=1)
    df_dataset["count"] = df_dataset["count"].astype(int)
    df_counter = df_dataset.groupby("pit").max()["count"]
    df_counter = df_counter.apply(lambda x: x + 1)
    name_counter = defaultdict(int, df_counter.to_dict())

dateparser = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')

# Load dataset metadata
df1 = pd.read_csv(os.path.join(SOURCE_DIR, FILENAME1),
                 encoding='utf-8',
                 sep=';',
                 parse_dates=['date'],
                 date_parser=dateparser)

df2 = pd.read_csv(os.path.join(SOURCE_DIR, FILENAME2),
                 encoding='latin-1',
                 sep=';',
                 parse_dates=['date'],
                 date_parser=dateparser)
df2 = df2.assign(dataset='Pier', Period='0')

df = pd.concat([df1, df2], ignore_index=True)

# Preprocessing of metadata and removal of erroneous data
df['pit'] = pd.to_numeric(df['pit'], errors='coerce')
df = df[df['pit'].apply(lambda x: not pd.isna(x))]
df['pit'] = df['pit'].astype(int)
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
pit_count = df['pit'].value_counts().to_dict()
df['pit_occurance'] = df.apply(lambda row: pit_count[row['pit']], axis=1)
df = df[df['pit_occurance'] >= 2]

# Mapping from period and dataset to folder structure
dataset_map = {
    ("LabMain", "Oct"): "labOct2019",
    ("LabMain", "May"): "labMai2020",
    ("FieldMain", "2"): "Periode2 (P1 2018)",
    ("FieldMain", "2,5"): "Periode2.5 (ekstramerking 2018)",
    ("FieldMain", "3"): "Periode3 (P2 2018)",
    ("FieldMain", "4"): "Periode4 (P3 2018)",
    ("FieldMain", "8"): "Periode8 (P3 2019)",
    ("FieldMain", "9"): "Periode9 (P1 2020)",
    ("FieldMain", "10"): "Periode10 (P2 2020)",
    ("FieldMain", "11"): "Periode11 (P3 2020)",
    ("Nedfisk", "10,5"): "Nedfisking_Bleikjo",
    ("Pier", "0"): "PierRecaps"
}

error_indices = []
error_names = []
success_indices = []
old_names = []
new_names = []

error_count = 0
success_count = 0

# Go through all fields in dataset
for index, row in df.iterrows():

    # Skip if record already exists in database
    if dataset_exists:
        if int(row["løpernummer"]) in df_dataset["løpernummer"].values:
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
        full_name = False
        prefix = row['prefix']
        zfill_num = 0
        if row['date'] == pd.Timestamp(2018, 5, 11):
            month = ""
            day = ""
            #zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 11.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 12):
            #prefix = "_11"
            month = ""
            day = ""
            #zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 12.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 13):
            #prefix = "_11"
            month = ""
            day = ""
            #zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 13.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 14):
            #prefix = "_11"
            month = ""
            day = ""
            #zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 14.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 15):
            #prefix = "_11"
            month = ""
            day = ""
            #zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 15.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 16):
            #prefix = "_"

            # Looks like these should work, but are omitted because it's not certain
            #prefix = "_1161"
            #prefix = row['prefix']
            month = ""
            day = ""
            #zfill_num = 5

            dataset_path = os.path.join(dataset_path, 'Bilder 16.5.18')
        elif row['date'] == pd.Timestamp(2018, 5, 17):
            #prefix = "_11"
            month = ""
            day = ""
            #zfill_num = 5
            dataset_path = os.path.join(dataset_path, 'Bilder 17.5.18')

    # Exceptions
    if value == 'Periode2.5 (ekstramerking 2018)':
        full_name = False
        if row['prefix'] == "P5300":
            dataset_path = os.path.join(dataset_path, '20180530')
        elif row['prefix'] == "P5310":
            dataset_path = os.path.join(dataset_path, '20180531')
        elif row['prefix'] == "P6010":
            dataset_path = os.path.join(dataset_path, '20180601')
        elif row['prefix'] == "P6030":
            dataset_path = os.path.join(dataset_path, '20180603')

    if value == 'PierRecaps':
        prefix = row['prefix'][0]
        if row['month'] == 10:
            month = 'A'
        zfill_num = 4

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

            # Keep track of error indices, but ignore this specific date
            if row['date'] != pd.Timestamp(2018, 5, 11):
                error_indices.append(index)
                error_names.append(file_name)

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

            if COPY_FILES:
                shutil.copy2(file_path, out_path)

            success_indices.append(index)
            old_names.append(file_name)
            new_names.append(out_filename)
            success_count += 1

# Display the number of successes
print(str(success_count) + "/" + str(error_count + success_count))

if SAVE_ERRORS:
    df_error = df.iloc[error_indices]
    df_error['filename'] = error_names
    df_error.to_csv("extraction_errors.csv", index=False, encoding='latin-1')

if WRITE_FILES:
    df_success = df.loc[success_indices]
    df_success['old_filename'] = old_names
    df_success['new_filename'] = new_names
    df_success.to_csv(os.path.join(DEST_DIR, "dataset.csv"), index=False, encoding='latin-1')

    # with open(os.path.join(DEST_DIR, CLASSES_FILE), "w") as classes_file:
    #     for key in name_counter.keys():
    #         classes_file.write(str(key) + "\n")
