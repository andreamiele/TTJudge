#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: backgroundSubstraction.py
# Created: Monday, 13th March 2023 10:29:07 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Wednesday, 15th March 2023 2:35:52 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023
#
#  ==============================================================================
import os
import sys
from os.path import abspath, dirname

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Functions.loaders import load_json, load_labels


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class BallDataset(Dataset):
    def __init__(self, train=True):
        return self


class CNNBlock(nn.Module):
    def __init__(self, train=True):
        return self


class ResidualBlock(nn.Module):
    def __init__(self, train=True):
        return self


class BallYOLOv3(nn.Module):
    def __init__(self, train=True):
        return self


class BallLocator:
    def __init__(self, train=True):
        return self


if __name__ == "__main__":
    x = BallLocator()
    x.run()
