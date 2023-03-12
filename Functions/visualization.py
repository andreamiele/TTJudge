#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: visualization.py
# Created: Sunday, 12th March 2023 10:14:22 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# GitHub: https://www.github.com/andreamiele
# -----
# Last Modified: Sunday, 12th March 2023 10:15:41 am
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
