from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import cv2

from keras.api.keras import models, layers, activations, initializers, losses, optimizers, metrics, callbacks
from keras import mixed_precision

from tqdm import tqdm
import os

import itertools
import matplotlib.pyplot as plt


class DataModule:
    def __init__(self, root_path=None):
        if root_path[-1] == "/":
            root_path = root_path[:-1]
        self.ROOT_PATH = root_path

    def img_to_arr(self, width, height) -> np.ndarray:
        imgs = list()
        for fname in tqdm(os.listdir(f"{self.ROOT_PATH}/images/"), desc="img_to_arr"):
            img = cv2.imread(f"{self.ROOT_PATH}/images/{fname}")
            img = cv2.resize(src=img, dsize=(width, height))
            imgs.append(img)

        return np.array(imgs)

    def label_to_arr(self):
        csv_df = pd.read_csv(f"{self.ROOT_PATH}/metadata.csv")

        ord_enc = OrdinalEncoder()
        ord_output = ord_enc.fit_transform(np.expand_dims(csv_df["dx"].to_numpy(), axis=-1))

        return ord_output

    def label_to_onehot(self, ord_label):
        onehot_enc = OneHotEncoder()
        onehot_output = onehot_enc.fit_transform(ord_label)
        onehot_output = onehot_output.toarray()

        return onehot_output

    def ros(self, imgs, labels, w=None, h=None, c=None):
        oversample = RandomOverSampler()
        if len(imgs.shape) > 2:
            org_shape = imgs.shape
            imgs = np.reshape(imgs, newshape=(org_shape[0], -1))
            imgs, labels = oversample.fit_resample(imgs, labels)
            imgs = np.reshape(imgs, newshape=(-1,) + org_shape[1:])

            return imgs, labels
        else:
            if (w is None) or (h is None) or (c is None):
                raise ValueError("width, height, channel must be not None")
            imgs, labels = oversample.fit_resample(imgs, labels)
            imgs, labels = np.array(imgs), np.array(labels)
            imgs = np.reshape(imgs, newshape=(-1, h, w, c))

            return imgs, labels


class TrainModule:
    def __init__(self, batch_size, ckpt_path=None, model_path=None, log_dir=None):
        self.BATCH_SIZE = batch_size
        self.CKPT_PATH = ckpt_path
        if self.CKPT_PATH is None:
            print("WARNING: CKPT_PATH is None")
        self.MODEL_PATH = model_path
        if self.MODEL_PATH is None:
            print("WARNING: MODEL_PATH is None")
        self.LOG_DIR = log_dir
        if self.LOG_DIR is None:
            print("WARNING: LOG_DIR is None")

    def create_model(self, input_shape, num_conv_blocks, num_conv_in_blocks, num_cls):
        mixed_precision.set_global_policy("mixed_float16")
        input_tensor = layers.Input(input_shape, name="img")

        rescaling_layer = layers.experimental.preprocessing.Rescaling(scale=1 / 255.0)(input_tensor)

        x = rescaling_layer
        block_cnt = 1
        for i in range(1, (num_conv_blocks + 1)):
            num_conv_filters = 2 ** (4 + block_cnt)
            x_ = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                               kernel_initializer=initializers.he_normal(), name=f"conv2d_{i}_1")(x)
            x = layers.BatchNormalization(name=f"bn_{i}_1")(x_)

            for j in range(2, (num_conv_in_blocks + 1)):
                x = layers.Conv2D(filters=num_conv_filters, kernel_size=(3, 3), padding="same", activation=activations.relu if i <= 2 else activations.selu,
                                  kernel_initializer=initializers.he_normal(), name=f"conv2d_{i}_{j}")(x)
                x = layers.BatchNormalization(name=f"bn_{i}_{j}")(x)
            x = layers.Add()([x_, x])
            if i == num_conv_blocks:
                x = layers.AvgPool2D(padding="same", strides=(1, 1), name=f"avg_pool_2d")(x)
                break
            x = layers.MaxPooling2D(padding="same", name=f"mp_{i}")(x)

            if i % 2 == 0:
                block_cnt += 1

        x = layers.Flatten()(x)

        x = layers.Dense(1024, activation=activations.selu,
                         kernel_initializer=initializers.he_normal(), name="dense_1")(x)

        cls_out = layers.Dense(num_cls, activation=activations.softmax,
                               kernel_initializer=initializers.glorot_normal(), name="cls_out")(x)

        model = models.Model(input_tensor, cls_out)
        model.compile(
            optimizer=optimizers.Adam(),
            loss={
                "cls_out": losses.categorical_crossentropy
            },
            metrics={
                "cls_out": metrics.categorical_accuracy
            },
            run_eagerly=True
        )

        return model

    def training(self, model: models.Model, x_train, x_valid, y_train, y_valid):
        model.fit(
            x={"img": x_train}, y={"cls_out": y_train},
            batch_size=self.BATCH_SIZE,
            epochs=1000,
            verbose=1,
            callbacks=[
                callbacks.TensorBoard(
                    log_dir=self.LOG_DIR
                ),
                callbacks.ReduceLROnPlateau(
                    factor=0.5,
                    patience=3,
                    verbose=1,
                    min_lr=1e-8
                ),
                callbacks.ModelCheckpoint(
                    filepath=self.CKPT_PATH,
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=True
                ),
                callbacks.EarlyStopping(
                    min_delta=1e-5,
                    patience=11,
                    verbose=1
                )
            ],
            validation_data=(
                {"img": x_valid}, {"cls_out": y_valid}
            )
        )
        model.load_weights(self.CKPT_PATH)
        model.save(self.MODEL_PATH)


def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix', fname=None):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if fname is not None:
        plt.savefig(fname, dpi=300)
    plt.show()
