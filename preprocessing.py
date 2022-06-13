from utils import DataModule

from sklearn.model_selection import train_test_split
import numpy as np

ROOT_PATH = "D:/AI/data/HAM10000"
COMPRESSED = False
# recommended ratio: 4:3
W, H = 384, 288
TEST_SIZE = 0.3
ROS = True

dm = DataModule(root_path=ROOT_PATH)

imgs = dm.img_to_arr(width=W, height=H)
print(imgs.shape, imgs.dtype)

labels = dm.label_to_arr()
print(labels.shape)

if ROS:
    imgs, labels = dm.ros(imgs, labels)

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=TEST_SIZE)
print(
    x_train.shape,
    x_test.shape,
    y_train.shape,
    y_test.shape
)

if COMPRESSED:
    np.savez_compressed(file=f"npz/ham10000_{W}x{H}_{ROS}_compressed.npz",
                        x_train=x_train,
                        x_test=x_test,
                        y_train=y_train,
                        y_test=y_test)
else:
    np.savez(file=f"npz/ham10000_{W}x{H}_{ROS}.npz",
             x_train=x_train,
             x_test=x_test,
             y_train=y_train,
             y_test=y_test)