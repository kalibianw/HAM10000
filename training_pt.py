from utils_pt import CustomImageDataset, Model, ANNModule

from torch.utils.data import DataLoader
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
NUM_EPOCHS = 5

if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_CSV_FILE_PATH)
    train_cid = CustomImageDataset(
        root_path=os.path.dirname(TRAIN_CSV_FILE_PATH),
        img_names=train_df["image_names"].to_numpy(),
        labels=np.expand_dims(train_df["labels"].to_numpy(), axis=-1)
    )
    train_loader = DataLoader(train_cid, batch_size=BATCH_SIZE, shuffle=True)
    train_total_batch_size = math.ceil(len(train_df["labels"]) / 32)

    test_df = pd.read_csv(TEST_CSV_FILE_PATH)
    test_cid = CustomImageDataset(
        root_path=os.path.dirname(TEST_CSV_FILE_PATH),
        img_names=test_df["image_names"].to_numpy(),
        labels=np.expand_dims(test_df["labels"].to_numpy, axis=-1)
    )
    test_loader = DataLoader(test_cid, batch_size=BATCH_SIZE, shuffle=True)
    test_total_batch_size = math.ceil(len(test_df["labels"]) / 32)

    model = Model().to(DEVICE)

    summary(model, (32, 3, 288, 384))
    dummy_data = torch.empty((32, 3, 288, 384), device=DEVICE)
    dummy_data.to(DEVICE)
    torch.onnx.export(
        model,
        dummy_data,
        "model/pt.onnx"
    )

    optimizer = torch.optim.Adam(model.parameters())
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
