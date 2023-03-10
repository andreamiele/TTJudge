# ==============================================================================
# File: download_data.py
# Project: src
# Author: Andrea Miele
# -----
# Last Modified:
# Modified By:
# -----
#
# -----
# Downloading all data from the OpenTTGames Dataset: https://lab.osai.ai/datasets/openttgames/
# ==============================================================================

import os
import sys
import wget
import zipfile
from os.path import abspath, dirname

R_PATH = dirname(dirname(abspath(__file__)))
if R_PATH not in sys.path:
    sys.path.append(R_PATH)


class DataDownload:
    def __init__(self):
        pass

    def folderscreate(self):  # Top Level
        """
        makes a list of all the paths that need to be constructed, then builds them if none already exist
        """
        data = R_PATH + "/Data"
        train = R_PATH + "/Data/Train"
        test = R_PATH + "/Data/Test"
        trains = [R_PATH + f"/Data/Train/Game{i+1}/" for i in range(5)]
        tests = [R_PATH + f"/Data/Test/Game{i+1}/" for i in range(7)]
        all = [data, train, test] + trains + tests
        for path in all:
            if not os.path.exists(path):
                os.mkdir(path)
        print("Task done! Folders created!")

    # TODO
    #  s / check if already downloaded / download zip and unzip and delete zip / download all other data files (json etc.)


    def videosdownload(self):  # Top Level
        """
        downloads all the videos of ping pong gameplay to their folders
        """
        test_paths = [f'https://lab.osai.ai/datasets/openttgames/data/test_{i+1}.mp4' for i in range(7)]
        self.listdownload(test_paths, file_type='Test')
        train_paths = [f'https://lab.osai.ai/datasets/openttgames/data/game_{i+1}.mp4' for i in range(5)]
        self.listdownload(train_paths, file_type='Train')
    
    def listdownload(self, path_list, file_type='Test'):  # Specific Helper download_videos
        """
        given a list of mp4 url paths, this uses wget to download them and put them in a destination path
        """
        for i, path in enumerate(path_list):
            dest_path = R_PATH + f"/Data/{file_type}/Game{i+1}/gameplay.mp4"
            if not os.path.exists(dest_path):
                print(f"I'm downloading {path} to {dest_path}...")
                wget.download(path, out=dest_path)

    def run(self):  # Run
        self.folderscreate()
        self.videosdownload()
        self.markupsdownload()

if __name__ == '__main__':
    x = DataDownload()
    self = x
    x.run()