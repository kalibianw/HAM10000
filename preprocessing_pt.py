from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import os

CSV_FILE_PATH = "D:/AI/data/HAM10000/metadata.csv"
if os.path.exists(CSV_FILE_PATH) is False:
    print("File doesn't exist.")
TRAIN_CSV_EXPORT_PATH = "D:/AI/data/HAM10000/train.csv"
TEST_CSV_EXPORT_PATH = "D:/AI/data/HAM10000/test.csv"
TEST_SIZE = 0.3

if __name__ == '__main__':
    org_df = pd.read_csv(CSV_FILE_PATH)
    print(org_df)

    image_paths = org_df["image_id"].to_numpy()
    labels = org_df["dx"].to_numpy()

    ord_enc = LabelEncoder()

    ros = RandomOverSampler()
    image_paths, labels = ros.fit_resample(
        X=np.expand_dims(image_paths, axis=-1),
        y=np.expand_dims(labels, axis=-1)
    )

    labels = ord_enc.fit_transform(labels).flatten()
    print(image_paths.shape, labels.shape)
    print(type(image_paths), type(labels))

    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=TEST_SIZE)
    print(
        train_image_paths.shape,
        test_image_paths.shape,
        train_labels.shape,
        test_labels.shape
    )

    train_df = pd.DataFrame(
        {
            "image_paths": train_image_paths.flatten(),
            "labels": train_labels
        },

    )
    test_df = pd.DataFrame(
        {
            "image_paths": test_image_paths.flatten(),
            "labels": test_labels
        }
    )

    train_df.to_csv(TRAIN_CSV_EXPORT_PATH, index=False)
    test_df.to_csv(TEST_CSV_EXPORT_PATH, index=False)
