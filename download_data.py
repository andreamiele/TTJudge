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

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
