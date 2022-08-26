import numpy as np

from utils import Generator, TrainModule

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from model_profiler import model_profiler
import tensorflow as tf

import shutil
import os


def mkdir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def rmkdir(path):
    if os.path.exists(os.path.dirname(path)):
        shutil.rmtree(os.path.dirname(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


def read_csv(root_path, csv_file_name, is_one_hot):
    df = pd.read_csv(os.path.join(root_path, csv_file_name))
    image_names = df["image_names"]
    labels = df["labels"]
    if is_one_hot:
        ohe = OneHotEncoder()
        labels = ohe.fit_transform(np.expand_dims(labels, axis=-1)).toarray()

    return image_names, labels


ROOT_PATH = "D:/AI/data/HAM10000/"
CKPT_PATH = rmkdir("ckpt/keras_training/keras_training.ckpt")
MODEL_PATH = mkdir("model/keras_model.h5")
LOG_DIR = rmkdir("log/keras_training/")
BATCH_SIZE = 32
VALID_SIZE = 0.2
NUM_CONV_BLOCKS = 8
NUM_CONV_IN_BLOCKS = 3

if __name__ == '__main__':
    tm = TrainModule(batch_size=BATCH_SIZE,
                     ckpt_path=CKPT_PATH,
                     model_path=MODEL_PATH,
                     log_dir=LOG_DIR)

    train_image_names, train_labels = read_csv(root_path=ROOT_PATH, csv_file_name="train.csv", is_one_hot=True)
    train_generator = Generator(
        root_path=ROOT_PATH,
        img_names=train_image_names,
        labels=train_labels
    )
    train_dataset = tf.data.Dataset.from_generator(
        generator=train_generator.get_data_generator,
        output_types=(tf.float32, tf.float32)
    )
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)

    valid_image_names, valid_labels = read_csv(root_path=ROOT_PATH, csv_file_name="valid.csv", is_one_hot=True)
    valid_generator = Generator(
        root_path=ROOT_PATH,
        img_names=train_image_names,
        labels=train_labels
    )
    valid_dataset = tf.data.Dataset.from_generator(
        generator=train_generator.get_data_generator,
        output_types=(tf.float32, tf.float32)
    )
    valid_dataset = valid_dataset.batch(batch_size=BATCH_SIZE)

    model = tm.create_model(
        input_shape=(288, 384, 3),
        num_conv_blocks=NUM_CONV_BLOCKS,
        num_conv_in_blocks=NUM_CONV_IN_BLOCKS,
        num_cls=7
    )
    model.save(filepath=MODEL_PATH)
    model.summary()
    model_profiler(model=model, Batch_size=BATCH_SIZE, verbose=1)

    tm.training(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset
    )
