#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: loaders.py
# Created: Friday, 10th March 2023 2:23:43 pm
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Saturday, 11th March 2023 1:50:06 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023
#
#  ==============================================================================

import os
from os.path import abspath, dirname
import pickle
import sys
import json

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_game_folders(train=True, test=True):
    x = []
    if train:
        train_path = ROOT_PATH + "/Data/Train"
        train_game_paths = listdir_fullpath(train_path)
        x += train_game_paths
    if test:
        test_path = ROOT_PATH + "/Data/Test"
        test_game_paths = listdir_fullpath(test_path)
        x += test_game_paths
    return x
