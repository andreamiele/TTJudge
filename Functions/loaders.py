#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: loaders.py
# Created: Friday, 10th March 2023 2:23:43 pm
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Saturday, 11th March 2023 1:46:59 pm
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


def load_json(json_path):  # Runs
    """
    simple json load
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data
