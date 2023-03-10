#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: loaders.py
# Created: Friday, 10th March 2023 2:23:43 pm
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Saturday, 11th March 2023 9:19:45 pm
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
import cv2

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def temporaryClearing():
    fld = ROOT_PATH + "/Temp/"
    f = listdir_fullpath(fld)
    for x in f:
        os.remove(x)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_games(train=True, test=True):
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


def load_labels(train=True, test=True):  # Run
    x = []
    game_folders = load_games(train=train, test=test)
    for item in game_folders:
        label_paths = [
            file
            for file in listdir_fullpath(item)
            if file.endswith(".json") and "predictions" not in file
        ]
        x += label_paths

    return x


def videoStreamLoader(path):
    capture = cv2.VideoCapture(path)
    nb_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stream = CamGear(source=path).start()
    return nb_frames, stream


def load_frames(path):
    frames_path = path.replace(".json", "_frames/")
    frames = sorted(
        listdir_fullpath(frames_path), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    nb = len(frames)
    tab = []
    for i in range(4, nb - 4):
        group = frames[i - 4 : i + 5]
        tab.append(group)

    return tab
