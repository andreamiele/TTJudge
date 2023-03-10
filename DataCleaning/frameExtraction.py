#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: frameExtraction.py
# Created: Friday, 10th March 2023 11:49:26 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 10th March 2023 1:37:58 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
# 
# -----
# Copyright (c) 2023
# 
#  ==============================================================================


import os
import random
import sys
from os.path import abspath, dirname
import cv2
from tqdm import tqdm
from vidgear.gears import CamGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]