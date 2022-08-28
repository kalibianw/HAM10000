from utils_pt import CustomImageDatasetLoadAllIntoMemory, Model, ANNModule

from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

import os
import time
import math

TRAIN_CSV_FILE_PATH = "D:/AI/data/HAM10000/train.csv"
if os.path.exists(TRAIN_CSV_FILE_PATH) is False:
    raise FileNotFoundError("Train csv file doesn't exists.")
TEST_CSV_FILE_PATH = "D:/AI/data/HAM10000/test.csv"
if os.path.exists(TEST_CSV_FILE_PATH) is False:
    raise FileNotFoundError("Test csv file doesn't exists.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch version: {torch.__version__} - Device: {DEVICE}")
BATCH_SIZE = 32
NUM_EPOCHS = 100
QUANTITY_EACH_LABELS = -1
DSIZE = (288, 384)


def read_csv(csv_file_path, quantity_each_labels=-1):
    df = pd.read_csv(csv_file_path)
    image_names, labels = (df["image_names"].to_numpy(), df["labels"].to_numpy()) if quantity_each_labels < 0 else extract_small(df, num_each_label=quantity_each_labels)
    total_batch_size = math.ceil(len(labels) / BATCH_SIZE)
    cid = CustomImageDatasetLoadAllIntoMemory(
        root_path=os.path.dirname(csv_file_path),
        image_names=image_names,
        labels=np.expand_dims(labels, axis=-1),
        pre_transform=torch.nn.Sequential(
            transforms.Resize(DSIZE)
        )
    )
    loader = DataLoader(cid, batch_size=BATCH_SIZE, shuffle=True)

    return total_batch_size, loader


def extract_small(df: pd.DataFrame, num_each_label):
    new_image_names = list()
    new_labels = list()
    for unique_key in df["labels"].unique():
        indices = df.index[df["labels"] == unique_key].tolist()
        for index in indices[:num_each_label]:
            new_image_names.append(df["image_names"].iloc[index])
            new_labels.append(df["labels"].iloc[index])

    return np.array(new_image_names), np.array(new_labels)


if __name__ == '__main__':
    train_total_batch_size, train_loader = read_csv(TRAIN_CSV_FILE_PATH, quantity_each_labels=QUANTITY_EACH_LABELS)
    test_total_batch_size, test_loader = read_csv(TEST_CSV_FILE_PATH, quantity_each_labels=QUANTITY_EACH_LABELS)

    model = Model().to(DEVICE)

    summary(model, (32, 3, 288, 384))
    dummy_data = torch.empty((32, 3, 288, 384), device=DEVICE)
    dummy_data.to(DEVICE)
    torch.onnx.export(
        model,
        dummy_data,
        "model/pt_empty.onnx"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    tm = ANNModule(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=32,
        device=DEVICE
    )

    local_time = time.localtime()
    log_path = f"log/{local_time[0]}_{local_time[1]}_{local_time[2]}_{local_time[3]}_{local_time[4]}_{local_time[5]}"
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_acc, train_loss = tm.train(train_loader, epoch_cnt=epoch, total_batch_size=train_total_batch_size)
        print(f"\n[EPOCH: {epoch}] - Train Loss: {train_loss:.4f}; Train Accuracy: {train_acc:.2f}%")

        test_acc, test_loss = tm.evaluate(test_loader, total_batch_size=test_total_batch_size)
        print(f"\n[EPOCH: {epoch}] - Test Loss: {test_loss:.4f}; Test Accuracy: {test_acc:.2f}%")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
    print(time.time() - start_time)
