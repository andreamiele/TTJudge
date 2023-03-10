#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: train_table_detection.py
# Created: Friday, 10th March 2023 11:47:53 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 10th March 2023 2:33:59 pm
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

#Import utility functions
from Functions.images import hflip, gaussblur, colorjitter


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]