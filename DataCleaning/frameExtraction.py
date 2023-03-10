#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: frameExtraction.py
# Created: Friday, 10th March 2023 11:49:26 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 10th March 2023 1:41:45 pm
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

class FrameFolders:
    def __init__(self):
        self.train_games = listdir_fullpath(ROOT_PATH + "/Data/Train/")
        self.test_games = listdir_fullpath(ROOT_PATH + "/Data/Test/")
        self.game_folders = self.train_games + self.test_games
        
    def run(self, n=3000):  # Run
        self.folderscreate()
        vid_paths = self.load_vid_paths()
        for i, vid_path in enumerate(vid_paths):
            print(i, vid_path)
            self.save_frames(vid_path, n)



if __name__ == '__main__':
    x = FrameFolders()
    self = x
    x.run()