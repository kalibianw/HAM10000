from utils import TrainModule

from model_profiler import model_profiler

from sklearn.model_selection import train_test_split
import numpy as np

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


NPZ_PATH = f"npz/ham10000_512x384.npz"
NPZ_FNAME = os.path.splitext(os.path.basename(NPZ_PATH))[0]
CKPT_PATH = rmkdir(f"ckpt/{NPZ_FNAME}/{NPZ_FNAME}.ckpt")
MODEL_PATH = mkdir(f"model/{NPZ_FNAME}.h5")
LOG_DIR = rmkdir(f"log/{NPZ_FNAME}/")
BATCH_SIZE = 32
VALID_SIZE = 0.2
NUM_CONV_BLOCKS = 8
NUM_CONV_IN_BLOCKS = 3

tm = TrainModule(batch_size=BATCH_SIZE,
                 ckpt_path=CKPT_PATH,
                 model_path=MODEL_PATH,
                 log_dir=LOG_DIR)

npz_loader = np.load(file=NPZ_PATH)
x_train_all, x_test = npz_loader["x_train"], npz_loader["x_test"]
y_train_all, y_test = npz_loader["y_train"], npz_loader["y_test"]

x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=VALID_SIZE)
print(
    x_train.shape,
    x_valid.shape,
    x_test.shape,
    y_train.shape,
    y_valid.shape,
    y_test.shape
)

model = tm.create_model(input_shape=x_train.shape[1:],
                        num_conv_blocks=NUM_CONV_BLOCKS,
                        num_conv_in_blocks=NUM_CONV_IN_BLOCKS,
                        num_cls=y_train.shape[1])
model.save(filepath=MODEL_PATH)
model.summary()
model_profiler(model=model, Batch_size=BATCH_SIZE, verbose=1)

tm.training(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid
)
