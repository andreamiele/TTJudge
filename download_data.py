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
# You can also download the date here: https://www.kaggle.com/datasets/anshulmehtakaggl/table-tennis-games-dataset-ttnet
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


    def zipdownload(self, path_list, file_type='Test'):  # Specific Helper  download_markups
        """
        given a list of url paths to zip files, this will download the zip, unpack it in its appropriate game folder, then delete the zip
        """
        for i, path in enumerate(path_list):
            dest_folder = R_PATH + f"/Data/{file_type}/Game{i+1}/"
            dest_path = dest_folder + "markups.zip"
            if not self.markups_exist(dest_folder):
                print(f"I'm downloading {path} to {dest_path}...")
                wget.download(path, dest_path)
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_folder)
                os.remove(dest_path)
    def markups_exist(self, dest_folder):  # Helping Helper  _download_zip
        """
        determines if the markup files have been downloaded already in the destination path
        """
        dest_files = os.listdir(dest_folder)
        markup_files = ['events_markup.json', 'segmentation_masks', 'ball_markup.json']
        for file in markup_files:
            if file not in dest_files:
                return False
        return True

    def markupsdownload(self):  # Top Level
        """
        downloads all the markup files (jsons for ball location, events, and semantic segmentation files)
        """
        test_paths = [f'https://lab.osai.ai/datasets/openttgames/data/test_{i+1}.zip' for i in range(7)]
        self.zipdownload(test_paths, file_type='Test')
        train_paths = [f'https://lab.osai.ai/datasets/openttgames/data/game_{i+1}.zip' for i in range(5)]
        self.zipdownload(train_paths, file_type='Train')

    def run(self):  # Run
        self.folderscreate()
        self.videosdownload()
        self.markupsdownload()

if __name__ == '__main__':
    x = DataDownload()
    self = x
    x.run()

# TODO
# Refactor code, comments, change functions names and variable for some etc.