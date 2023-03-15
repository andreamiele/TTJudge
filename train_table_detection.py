#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: train_table_detection.py
# Created: Friday, 10th March 2023 11:47:53 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Wednesday, 15th March 2023 2:16:33 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023
#
#  ==============================================================================

import wandb

import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from skimage import draw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

import os
import random
import time
import sys
from os.path import abspath, dirname


ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

# Import utility functions
from Functions.images import hflip, gaussblur, colorjitter
from Functions.loaders import temporaryClearing, load_json


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class TableDataset(Dataset):
    def __init__(self, train):
        self.game_paths = (
            listdir_fullpath(ROOT_PATH + "/Data/Train/")
            if train
            else listdir_fullpath(ROOT_PATH + "/Data/Test/")
        )
        self.iPaths, self.corners = self.load_imgs_corners()
        self.train = train
        self.test = not train

    def load_imgs_corners(self):
        iPaths = []
        corners = []
        for path in self.game_paths:
            current_iPaths = listdir_fullpath(path + "/frames/")
            current_corners = load_json(path + "/table.json")
            iPaths += current_iPaths
            corners += [current_corners] * len(current_iPaths)
        x = list(zip(iPaths, corners))
        random.shuffle(x)
        iPaths, corners = zip(*x)
        return list(iPaths), list(corners)

    def cornersCleaner(self, corners):
        x = int(round(corners["x"] * (320 / 1920)))
        y = int(round(corners["y"] * (128 / 1080)))
        return np.array([y, x])

    def dataAugmentation(self, img, mask):
        randomNB = random.random()
        if randomNB < 0.2:
            img = colorjitter(img)
        elif randomNB < 0.4:
            img = hflip(img)
            mask = hflip(mask)
        elif randomNB < 0.6:
            img = gaussblur(img)
        return img, mask

    def __len__(self):
        return len(self.corners)

    def __getIndex__(self, index):
        iPath, corners = self.iPaths[index], self.corners[index]
        img = read_image(iPath).to("cuda") / 255.0
        img = transforms.Resize(size=(128, 320))(img).float()
        mask = self.maske(img, corners)
        img, mask = self.dataAugmentation(img, mask)
        img = transforms.Normalize([0, 0, 0], [1, 1, 1])(img)
        mask = transforms.Normalize([0], [1])(mask)
        return img, mask.float()

    def maske(self, img, corners):
        c1, c2, c3, c4 = (
            self.cornersCleaner(corners["Corner 1"]),
            self.cornersCleaner(corners["Corner 2"]),
            self.cornersCleaner(corners["Corner 3"]),
            self.cornersCleaner(corners["Corner 4"]),
        )
        polygon = np.array([c1, c2, c3, c4])
        height, width = img.shape[1], img.shape[2]
        mask = draw.polygon2mask((height, width), polygon)
        mask = mask.astype(int)
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0)
        return mask.to("cuda").float()


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConvolution(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConvolution(feature * 2, feature))
        self.bottleneck = DoubleConvolution(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)


class Train:
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConvolution(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConvolution(feature * 2, feature))
        self.bottleneck = DoubleConvolution(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def save_input(self, X, y, epoch, batch_idx):
        if torch.rand(1).item() > 0.99:
            for i in range(X.shape[0]):
                img = X[i]
                mask = y[i]
                mask = np.array(transforms.ToPILImage()(mask).convert("RGB"))
                arr = np.array(transforms.ToPILImage()(img).convert("RGB"))
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                arr = np.maximum(arr, mask)
                assert cv2.imwrite(
                    ROOT_PATH + f"/Temp/label_epoch_{epoch}_batch_{batch_idx}_{i}.png",
                    arr,
                )

    def save_pred(self, X, y, pred, epoch, batch_idx, test=False):
        if torch.rand(1).item() > 0.9:
            for i in range(X.shape[0]):
                mask = pred[i]
                img = y[i]
                mask = np.array(transforms.ToPILImage()(mask).convert("RGB"))
                arr = np.array(transforms.ToPILImage()(img).convert("RGB"))
                combo = np.hstack((arr, mask))
                test_str = "Test_" if test else ""
                assert cv2.imwrite(
                    ROOT_PATH
                    + f"/Temp/{test_str}pred_epoch_{epoch}_batch_{batch_idx}_{i}.png",
                    combo,
                )

    def training(self, epoch):
        self.model.train()

        for batch_idx, (X, y) in enumerate(self.train_dataloader):
            with torch.cuda.amp.autocast():
                self.save_input(X, y, epoch, batch_idx)
                pred = self.model(X)
                binary_pred = (pred > 0.5).float()
                self.save_pred(X, y, binary_pred, epoch, batch_idx)
                loss = self.loss(pred, y)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.wandb:
                wandb.log({"Train Loss": loss.item()})
            if batch_idx % 10 == 0:
                print(
                    f"Batch {batch_idx+1}/{len(self.train_dataloader)}: Loss: {loss.item()}"
                )
            if batch_idx % 150 == 0:
                self.test_loop(epoch)

    def testing(self, epoch):
        self.model.eval()
        num_correct = 0
        num_pixels = 0
        dice_score = 0

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.test_dataloader):
                preds = torch.sigmoid(self.model(X))
                loss = self.loss(preds, y)
                preds = (preds > 0.5).float()
                self.save_pred(X, y, preds, epoch, batch_idx, test=True)
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        if self.wandb:
            wandb.log(
                {
                    "Correct": 100 * (num_correct / num_pixels),
                    "Dice Score": dice_score / len(self.test_dataloader),
                    "Test Loss": loss,
                }
            )

        print(f"Test Loss: {loss.item()}")
        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        )
        print(f"Dice score: {dice_score/len(self.test_dataloader)}")
        return dice_score / len(self.test_dataloader)

    def run(self):
        temporaryClearing()
        max_dice = float("-inf")
        dice_dec_count = 0
        for i in range(self.epochs):
            print(f"Epoch {i+1}/{self.epochs}")
            self.train_loop(i)
            dice_score = self.test_loop(i)
            dice_dec_count = dice_dec_count + 1 if dice_score < max_dice else 0
            if dice_score > max_dice:
                print("<< Saving model ...>>")
                dice_str = "{:f}".format(dice_score).replace(".", "")
                lr_str = "{:f}".format(self.learning_rate)[2:]
                torch.save(
                    self.model.state_dict(),
                    ROOT_PATH
                    + f"/Modeling/Table_Segmentation_UNET_{dice_str}_bs_{self.batch_size}_lr_{lr_str}.pth",
                )
                max_dice = dice_score
            if dice_dec_count >= 5:
                print("<<Early stopping>>")


def wandbSweep():
    def trainingWandb(config=none):
        with wandb.init(config=config):
            config = wandb.config
            trainer = Train(
                batch_size=config.batch_size, learning_rate=config.learning_rate
            )
            trainer.run()

    sweepId = ""
    wandb.agent(sweepId, trainingWandb, count=100)


if __name__ == "__main__":
    x = Train()
    x.run()
    # wandbSweep()
