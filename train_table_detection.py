#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: train_table_detection.py
# Created: Friday, 10th March 2023 11:47:53 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Monday, 13th March 2023 9:56:54 am
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
