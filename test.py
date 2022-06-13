from keras.api.keras import models

from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix
import numpy as np
import os

NPZ_PATH = f"npz/ham10000_384x288_False.npz"
model_paths = [f"model/ham10000_384x288_False.h5", f"model/ham10000_384x288_True.h5"]
BATCH_SIZE = 32

npz_loader = np.load(NPZ_PATH)
x_test, y_test = npz_loader["x_test"], npz_loader["y_test"]
print(x_test.shape, y_test.shape)

for model_path in model_paths:
    model = models.load_model(model_path)

    result = model.predict(
        x={"img": x_test},
        verbose=1,
        batch_size=BATCH_SIZE
    )
    conf_mat = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(result, axis=1))
    print(conf_mat)
    plot_confusion_matrix(conf_mat,
                          target_names=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
                          normalize=True,
                          title=f"Confusion matrix of {os.path.basename(os.path.splitext(model_path)[0])}",
                          fname=f"{os.path.basename(os.path.splitext(model_path)[0])}.png")
    print("----------------------------------------")
