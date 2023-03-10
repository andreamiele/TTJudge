#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: images.py
# Created: Friday, 10th March 2023 2:24:00 pm
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 10th March 2023 2:32:06 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
# 
# -----
# Copyright (c) 2023
# 
#  ==============================================================================


import sys
import numpy as np
import cv2
from os.path import abspath, dirname

from torchvision import transforms

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def tensor_to_arr(tensor):
    tab = np.array(transforms.ToPILImage()(tensor).convert('RGB'))
    tab = cv2.cvtColor(tab, cv2.COLOR_BGR2RGB)
    return tab

def colorjitter(tensor):
    f = transforms.ColorJitter(brightness=0.5, hue=0.3)
    return f(tensor)


def gaussblur(tensor):
    f = transforms.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 5))
    return f(tensor)

def hflip(tensor):
    f = transforms.RandomHorizontalFlip(p=1)
    return f(tensor)


#TODO
# Add comments on the code