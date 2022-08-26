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
ROS = True

if __name__ == '__main__':
    org_df = pd.read_csv(CSV_FILE_PATH)
    print(org_df)

    image_names = org_df["image_id"].to_numpy()
    labels = org_df["dx"].to_numpy()

    ord_enc = LabelEncoder()

    if ROS:
        ros = RandomOverSampler()
        image_names, labels = ros.fit_resample(
            X=np.expand_dims(image_names, axis=-1),
            y=np.expand_dims(labels, axis=-1)
        )
        labels = labels.flatten()

    labels = ord_enc.fit_transform(labels)
    print(image_names.shape, labels.shape)
    print(type(image_names), type(labels))

    train_image_names, test_image_names, train_labels, test_labels = train_test_split(image_names, labels, test_size=TEST_SIZE)
    print(
        train_image_names.shape,
        test_image_names.shape,
        train_labels.shape,
        test_labels.shape
    )

    train_df = pd.DataFrame(
        {
            "image_names": train_image_names.flatten(),
            "labels": train_labels
        },

    )
    test_df = pd.DataFrame(
        {
            "image_names": test_image_names.flatten(),
            "labels": test_labels
        }
    )

    train_df.to_csv(TRAIN_CSV_EXPORT_PATH, index=False)
    test_df.to_csv(TEST_CSV_EXPORT_PATH, index=False)
