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
        print("Folders created!")

    # TODO
    # Download list of url path / download videos / check if already downloaded / download zip and unzip and delete zip / download all data files (json etc.)



if __name__ == '__main__':
    x = DataDownload()
    self = x
    x.run()