#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: judge.py
# Created: Friday, 17th March 2023 9:30:10 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 17th March 2023 9:32:40 am
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023
#
#  ==============================================================================
import sys
from os.path import abspath, dirname
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from vidgear.gears import WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Games.judgeHelper import JudgeHelper
from Functions.loaders import load_pickle, temporaryClearing
from Functions.visualization import showTable, showFrameNb


class Judge(JudgeHelper):
    def __init__(self) -> None:
        super().__init__()
