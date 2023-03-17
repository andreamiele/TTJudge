#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: judgeHelper.py
# Created: Friday, 17th March 2023 9:30:39 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 17th March 2023 9:36:37 am
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
import pickle
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Functions.frameReader import FrameReader
from Functions.loaders import temporaryClearing
from Functions.visualization import (
    drawContours,
    showArcD,
    showArcDC,
    showArcL,
    showBallBorder,
    showEventBox,
    showExtrArcC,
    showFrameNb,
    showTable,
)
