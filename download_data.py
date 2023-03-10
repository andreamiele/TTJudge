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
