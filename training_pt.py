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
import sys

TRAIN_CSV_FILE_PATH = "D:/AI/data/HAM10000/train.csv"
if os.path.exists(TRAIN_CSV_FILE_PATH) is False:
    raise FileNotFoundError("Train csv file doesn't exists.")
VALID_CSV_FILE_PATH = "D:/AI/data/HAM10000/valid.csv"
if os.path.exists(VALID_CSV_FILE_PATH) is False:
    raise FileNotFoundError("Valid csv file doesn't exists.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch version: {torch.__version__} - Device: {DEVICE}")
BATCH_SIZE = 32
NUM_EPOCHS = 100
QUANTITY_EACH_LABELS = -1
DSIZE = (288, 384)
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 3
REDUCE_LR_RATE = 0.5


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
    lr = 1e-3

    train_total_batch_size, train_loader = read_csv(TRAIN_CSV_FILE_PATH, quantity_each_labels=QUANTITY_EACH_LABELS)
    valid_total_batch_size, valid_loader = read_csv(VALID_CSV_FILE_PATH, quantity_each_labels=QUANTITY_EACH_LABELS)

    model = Model().to(DEVICE)

    summary(model, (32, 3, 288, 384))
    dummy_data = torch.empty((32, 3, 288, 384), device=DEVICE)
    dummy_data.to(DEVICE)
    torch.onnx.export(
        model,
        dummy_data,
        "model/pt_empty.onnx"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    tm = ANNModule(
        model=model,
        criterion=criterion,
        batch_size=32,
        device=DEVICE
    )

    local_time = time.localtime()
    log_path = f"log/{local_time[0]}_{local_time[1]}_{local_time[2]}_{local_time[3]}_{local_time[4]}_{local_time[5]}"
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    best_valid_loss = sys.float_info.max
    early_stopping_cnt = 0
    reduce_lr_cnt = 0
    start_time = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_acc, train_loss = tm.train(train_loader, optimizer=optimizer, epoch_cnt=epoch, total_batch_size=train_total_batch_size)
        print(f"\n[EPOCH: {epoch}] - Train Loss: {train_loss:.4f}; Train Accuracy: {train_acc:.2f}%")

        model, valid_acc, valid_loss = tm.evaluate(valid_loader, total_batch_size=valid_total_batch_size)
        print(f"\n[EPOCH: {epoch}] - Valid Loss: {valid_loss:.4f}; Valid Accuracy: {valid_acc:.2f}%")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/valid", valid_acc, epoch)
        writer.add_scalar("Learning rate/LR", lr, epoch)

        if valid_loss < best_valid_loss:
            early_stopping_cnt = 0
            reduce_lr_cnt = 0
            best_valid_loss = valid_loss
            torch.onnx.export(
                model,
                dummy_data,
                f"model/pt_[{valid_loss:.4f}].onnx"
            )
        else:
            early_stopping_cnt += 1
            reduce_lr_cnt += 1
            print(f"Valid loss didn't improved from {best_valid_loss}; current: {valid_loss}")
            print(f"Early stopping - {early_stopping_cnt} / {EARLY_STOPPING_PATIENCE}")
            if early_stopping_cnt >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
            if reduce_lr_cnt >= REDUCE_LR_PATIENCE:
                lr *= REDUCE_LR_RATE
                reduce_lr_cnt = 0
                print(f"Reduce LR triggered; Current learning rate: {lr}")
                continue
            print(f"Reduce LR - Current learning rate: {lr}; {reduce_lr_cnt} / {REDUCE_LR_PATIENCE}")

    print(f"Training time: {time.time() - start_time}s")
