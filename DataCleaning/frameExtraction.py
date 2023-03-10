#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: frameExtraction.py
# Created: Friday, 10th March 2023 11:49:26 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 10th March 2023 2:04:22 pm
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
    
    def folderscreate(self):  # Top Level
        """
        creating "frames" folders inside /Data/Train-or-Test/Game/ folders
        """
        for game_folder in self.game_folders:
            frame_folder = game_folder + "/frames"
            if not os.path.exists(frame_folder):
                os.mkdir(frame_folder)
                
    def load_vid_paths(self):  # Top Level
        """
        making a list of all Train/Test gameplay.mp4 paths
        """
        vid_paths = []
        for game_folder in self.game_folders:
            vid_path = game_folder + "/gameplay.mp4"
            assert os.path.exists(vid_path)
            vid_paths.append(vid_path)
        return vid_paths

    def save_frames(self, vid_path, n):  # Top Level
        """
        running through the video and saving n frames to the frames folder
        """
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stream = CamGear(source=vid_path).start()
        save_indices = random.sample(range(0, num_frames), n)
        for i in tqdm(range(num_frames)):
            frame = stream.read()
            if i in save_indices:
                save_path = vid_path.replace("gameplay.mp4", f"frames/{i}.png")
                assert cv2.imwrite(save_path, frame)
                
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