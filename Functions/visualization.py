#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: visualization.py
# Created: Sunday, 12th March 2023 10:14:22 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# GitHub: https://www.github.com/andreamiele
# -----
# Last Modified: Sunday, 12th March 2023 10:21:21 am
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023 Your Company
#
#  ==============================================================================

import cv2
import sys
import numpy as np
from os.path import abspath, dirname
from PIL import ImageFont, Image, ImageDraw

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def colorizeImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    return img


def drawContours(img, contours, color):
    img = img_to_color(img)
    img = cv2.drawContours(img, contours, -1, color, 3)
    return img


def contourCenter(contours):
    min_x, min_y, max_x, max_y = contourMaxMin(contours)
    return min_x + ((max_x - min_x) / 2), min_y + ((max_y - min_y) / 2)


def contourMaxMin(contours):
    min_x, min_y, max_x, max_y = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )
    for contour in contours:
        min_x = min(min_x, contour[:, :, 0].min())
        min_y = min(min_y, contour[:, :, 1].min())
        max_x = max(max_x, contour[:, :, 0].max())
        max_y = max(max_y, contour[:, :, 1].max())
    return min_x, min_y, max_x, max_y
