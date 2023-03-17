#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: judgeHelper.py
# Created: Friday, 17th March 2023 9:30:39 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 17th March 2023 10:05:16 am
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


class JudgeHelper:
    def __init__(self, start, end, saved):
        self.start = start
        self.end = end
        self.saved = saved

    def cleanDictionnary(self):
        data = {
            "Table": {},
            "All Contours": {},
            "Step 1 - Ball - Class": {},
            "Step 1 - Ball - Neigh": {},
            "Step 1 - Ball - Back": {},
            "Step 2 - Events": {},
            "Step 2 - Ball - Contours": {},
            "Step 2 - Ball - Centers": {},
            "Step 2 - Arcs": [],
            "Step 3 - Ball - Inter Event Centers": {},
            "Step 3 - Ball - Inter Arc Centers": {},
            "Step 3 - Ball - Ball Centers": {},
            "Step 4 - Events": {},
            "Step 4 - Arcs": [],
            "Step 4 - Contours": {},
        }
        return data

    def saveImgs():
        return None

    def runGame():
        return None


if __name__ == "__main__":
    saved = 0
    start = 0 - saved
    end = 0 - saved
    x = JudgeHelper(start, end, saved)
    self = x
    path = ROOT_PATH + "/Data/Train/xxx"
    x.runGame()
