#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: judgeHelper.py
# Created: Friday, 17th March 2023 9:30:39 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 17th March 2023 1:42:57 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023
#
#  ==============================================================================
import sys

sys.path.append("/usr/local/lib/python3.10.5/site-packages")
from os.path import abspath, dirname
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Functions.frameReader import FrameReader
from Functions.loaders import temporaryClearing
from Functions.visualization import (
    drawContours,
    showArcD,
    showArcDC,
    showArcL,
    showBallBorder,
    showEventBox,
    showExtrArcC,
    showFrameNb,
    showTable,
    contourCenter,
    contourMaxMin,
    wContour,
    contourDist,
)


class JudgeHelper:
    def __init__(self, start, end, saved):
        self.start = start
        self.end = end
        self.saved = saved

    def load_video(self, videoPath, loadFrames):  # Top Level
        if loadFrames:
            stream = FrameReader(start=self.frame_start, end=self.frame_end)
            nbFrame = len(stream)
        else:
            cap = cv2.VideoCapture(videoPath)
            nbFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stream = CamGear(source=videoPath).start()
        return stream, nbFrame

    def s1Ball(self, data, frame_idx):
        """
        finding the ball from step1, whether it's in classic/neighbor/backtracked
        - returns None if there is no ball
        """
        if frame_idx in data["Step 1 - Ball - Class"]:
            return data["Step 1 - Ball - Class"][frame_idx]
        elif frame_idx in data["Step 1 - Ball - Neigh"]:
            return data["Step 1 - Ball - Neigh"][frame_idx]
        elif frame_idx in data["Step 1 - Ball - Back"]:
            return data["Step 1 - Ball - Back"][frame_idx]
        return None

    def detectTable(self, output, frame, frame_idx):
        """
        detecting the table with segmentation inside the frame
        """
        # TODO run the actual segmentation model
        table = [1, 1, 1, 1, 1, 1, 1, 1]
        output["Table"][frame_idx] = table
        for i in range(frame_idx - 100, frame_idx):
            output["Table"][i] = table
        return output

    def findBall(self, data, previousFrame, frame, frameIndex):
        """
        using a frame and previous frame, this computes the difference between them and finds the ball using the classic, neighbor, and backtracking methods and updating 'data'
        """

    def frameDifference(self, pframe, frame):
        # Black and White images
        pframe = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian blur for the images
        pframe = cv2.GaussianBlur(pframe, (21, 21), 0)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        # Frames substraction
        difference = cv2.absdiff(pframe, frame)
        difference = cv2.threshold(difference, 7, 255, cv2.THRESH_BINARY)[1]
        difference = cv2.dilate(difference, None, iterations=2)
        return difference

    def findContours(self, diff):
        contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 6000 > cv2.contourArea(c) > 10]
        return contours

    def contoursList(self, contours):
        if not contours:
            return []
        contoursList = [[contours[0]]]
        contours = contours[1:]
        while len(contours) > 0:
            current_contour = contours.pop()
            added = False
            for i, contour_list in enumerate(contoursList):
                if contourDist(current_contour, contour_list[-1]) < 40 and (not added):
                    contoursList[i] = contour_list + [current_contour]
                    added = True
            if not added:
                contoursList.append([current_contour])
        return contoursList

    def frameDifferenceContours(self, frame1, frame2):

        diff = self.frameDifference(frame1, frame2)
        # ! temporarily blacking out kids in the background
        diff[336:535, 1080:1400] = 0
        diff[372:545, 360:660] = 0
        raw_contours = self.findContours(diff)
        contours = self.contoursList(raw_contours)
        return diff, contours

    def cleanDictionnary(self):
        data = {
            "Table": {},
            "All Contours": {},
            "Step 1 - Ball - Class": {},
            "Step 1 - Ball - Neigh": {},
            "Step 1 - Ball - Back": {},
            "Step 2 - Events": {},
            "Step 2 - Ball - Contours": {},
            "Step 2 - Ball - Centers": {},
            "Step 2 - Arcs": [],
            "Step 3 - Ball - Inter Event Centers": {},
            "Step 3 - Ball - Inter Arc Centers": {},
            "Step 3 - Ball - Ball Centers": {},
            "Step 4 - Events": {},
            "Step 4 - Arcs": [],
            "Step 4 - Contours": {},
        }
        return data

    def detect_hits(self, data):
        """ """

    def saveImgs(self, data, videoPath, load):  # TODO
        return None

    def runGame(self, videoPath, load, save=True):
        data = self.cleanDictionnary()

        if save:
            self.saveImgs(data, videoPath, load)
        return None


if __name__ == "__main__":
    saved = 0
    start = 0 - saved
    end = 0 - saved
    x = JudgeHelper(start, end, saved)
    self = x
    path = ROOT_PATH + "/Data/Train/xxx"
    load = True
    x.runGame()
