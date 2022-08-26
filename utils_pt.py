from opt_einsum.backends import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as tff

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from tqdm import tqdm
import os


class CustomImageDataset(Dataset):
    def __init__(self, root_path, img_paths, labels, dsize=(288, 384), transform=None):
        """
        :param dsize: (height, width)
        """
        self.root_path = root_path
        self.x_data = img_paths
        self.y_data = labels
        self.dsize = dsize
        self.transform = transform

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        x = read_image(os.path.join(self.root_path, f"images/{self.x_data[idx]}.jpg"))
        x = tff.resize(x, self.dsize)
        y = torch.Tensor(self.y_data[idx])

        if self.transform:
            x = self.transform(x)

        return x, y


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation_function="relu"):
        """
        :param activation_function: default: relu(support: relu, selu)
        """
        super(ResBlock, self).__init__()
        self.activation_function = activation_function
        if self.activation_function not in ["relu", "selu"]:
            raise AttributeError("Support activation function: relu or selu")

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        self.bn1 = nn.BatchNorm2d(
            num_features=out_channel
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=out_channel
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=out_channel
        )

    def forward(self, x):
        out_ = self.bn1(self.conv1(x))
        out = nnf.relu(out_) if self.activation_function == "relu" else nnf.selu(out_)

        out = self.bn2(self.conv2(out))
        out = nnf.relu(out) if self.activation_function == "relu" else nnf.selu(out)

        out = self.bn3(self.conv3(out))
        out += out_
        out = nnf.relu(out) if self.activation_function == "relu" else nnf.selu(out)

        return out


class Classifier(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.output = nn.Linear(
            in_features=hidden_features,
            out_features=num_classes
        )

    def forward(self, x):
        out = nnf.selu(self.linear1(x))
        out = nnf.softmax(self.output(out))

        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.res_block1 = ResBlock(
            in_channel=3,
            out_channel=32,
            activation_function="relu"
        )
        self.res_block2 = ResBlock(
            in_channel=32,
            out_channel=32,
            activation_function="relu"
        )
        self.res_block3 = ResBlock(
            in_channel=32,
            out_channel=64,
            activation_function="selu"
        )
        self.res_block4 = ResBlock(
            in_channel=64,
            out_channel=64,
            activation_function="selu"
        )
        self.res_block5 = ResBlock(
            in_channel=64,
            out_channel=128,
            activation_function="selu"
        )
        self.res_block6 = ResBlock(
            in_channel=128,
            out_channel=128,
            activation_function="selu"
        )
        self.res_block7 = ResBlock(
            in_channel=128,
            out_channel=256,
            activation_function="selu"
        )

        self.classifier = Classifier(
            in_features=3840,
            hidden_features=1024,
            num_classes=7
        )

    def forward(self, x):
        out = nnf.max_pool2d(self.res_block1(x), kernel_size=(2, 2))
        out = nnf.max_pool2d(self.res_block2(out), kernel_size=(2, 2))
        out = nnf.max_pool2d(self.res_block3(out), kernel_size=(2, 2))
        out = nnf.max_pool2d(self.res_block4(out), kernel_size=(2, 2))
        out = nnf.max_pool2d(self.res_block5(out), kernel_size=(2, 2))
        out = nnf.max_pool2d(self.res_block6(out), kernel_size=(2, 2))

        out = nnf.avg_pool2d(
            self.res_block7(out),
            kernel_size=(2, 2),
            stride=(1, 1)
        )

        out = torch.flatten(out, start_dim=1)

        out = self.classifier(out)

        return out


class ANNModule:
    def __init__(self, model, optimizer, criterion, batch_size, device):
        self.DEVICE = device
        self.BATCH_SIZE = batch_size

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_loader, epoch_cnt, total_batch_size):
        self.model.train()
        train_loss = 0
        correct = 0

        for batch_idx, (image, label) in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch_cnt}", total=total_batch_size):
            image = image.to(self.DEVICE)
            label = label.to(self.DEVICE)
            image = image.float()
            label = label.long()

            self.optimizer.zero_grad()

            output = self.model(image)
            loss = self.criterion(output, label.squeeze(dim=-1))
            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        train_loss /= (len(train_loader.dataset) / self.BATCH_SIZE)
        train_acc = 100. * correct / len(train_loader.dataset)

        return train_acc, train_loss

    def evaluate(self, test_loader, total_batch_size):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for image, label in tqdm(test_loader, desc="Evaluate...", total=total_batch_size):
                image = image.to(self.DEVICE)
                image = image.float
                label = label.to(self.DEVICE)
                label = label.long

                output = self.model(image)

                test_loss += self.criterion(output, label).item()
                prediction = output.max(1, keepdim=True)[1]
                correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= (len(test_loader.dataset) / self.BATCH_SIZE)
        test_acc = 100. * correct / len(test_loader.dataset)

        return test_acc, test_loss