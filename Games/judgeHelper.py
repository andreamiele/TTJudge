#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: judgeHelper.py
# Created: Friday, 17th March 2023 9:30:39 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Sunday, 19th March 2023 3:46:24 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023
#
#  ==============================================================================
import sys
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

    def findBallNeigh(self, frame_1_ball, contours):
        """
        locating the ball in the frame2 contours, based on the location of the ball in frame1
        - the ball in frame2 must be close to the ball in frame1
        """
        f1_min_x, f1_min_y, f1_max_x, _ = contourMaxMin(frame_1_ball)
        found = []
        for contour in contours:
            min_x, min_y, max_x, _ = contourMaxMin(contour)
            left = abs(f1_min_x - min_x) < 50
            right = abs(f1_max_x - max_x) < 50
            top_bottom = abs(f1_min_y - min_y) < 25
            if (left or right) and top_bottom and wContour(contour):
                found.append(contour)
        #
        return (
            min(found, key=lambda x: contourMaxMin((x)[3])) if len(found) > 0 else None
        )

    def areaClassic(self, countourList):
        area = sum([cv2.contourArea(contour) for c in contourList])
        return 50 < area < 3000

    def ballInTheMiddle(self, countourList, table):
        centers = [contourCenter(contour) for c in contourList]
        x = sum([c[0] for c in centers]) / len(centers)
        return (x > table[1] + 300) and (x < table[-1] - 300)

    def minimumDistance(self, ball, otherCountours):
        minimum = float("inf")
        others = [subitem for item in otherCountours for subitem in item]
        for x in others:
            distance = contourDist(self, ball[0], x)
            minimum = min(minimum, distance)
        return minimum

    def findBallClass(self, table, contours):
        """
        finding the ball in the frame the "classic" way: the ball must be towards the middle of the table,
        with only one contour match (the ball) and a good distance from any other movement
        """
        ballIndexs = []
        for i, contour in enumerate(contours):
            area = self.areaClassic(contour)
            localisation = self.ballInTheMiddle(contour, table)
            if localisation and area:
                ballIndexs.append(i)
        # TODO: check the distance
        if len(ballIndexs) == 1:  # We only have one item
            ball = contours[ballIndexs[0]]
            nonBalls = [c for i, c in enumerate(contours) if i != ballIndexs[0]]
            if self.minimumDistance(ball, nonBalls) > 300 and wContour(self, ball):
                return ball
        return None  # Else no balls

    def removeNetContours(self, data, contours, frame_idx):
        table = data["Table"][frame_idx]
        table_middle_x = table[1] + ((table[-1] - table[1]) / 2)
        new = []
        for contour in contours:
            contour_center = contourCenter((contour)[0])
            if abs(contour_center - table_middle_x) > 75:
                new.append(contour)
        return new

    def findBall(self, data, previousFrame, frame, frameIndex):
        """
        using a frame and previous frame, this computes the difference between them and finds the ball using the classic, neighbor, and backtracking methods and updating 'data'
        """
        diff, contours = self.frameDifferenceContours(previousFrame, frame)
        data["All Contours"][frame_idx] = contours
        contours = self.removeNetContours(data, contours, frame_idx)
        table = data["Table"][frame_idx]
        prev_ball_contour = (
            data["Step 1 - Ball - Class"][frame_idx - 1]
            if frame_idx - 1 in data["Step 1 - Ball - Class"]
            else None
        )
        prev_ball_contour = (
            data["Step 1 - Ball - Neigh"][frame_idx - 1]
            if frame_idx - 1 in data["Step 1 - Ball - Neigh"]
            and prev_ball_contour is None
            else prev_ball_contour
        )
        classic = False
        ball = (
            None
            if prev_ball_contour is None
            else self.findBallNeigh(prev_ball_contour, contours)
        )
        if ball is None:
            classic = True
            ball = self.findBallClass(table, contours)
        if ball is not None:
            key = "Step 1 - Ball - Class" if classic else "Step 1 - Ball - Class"
            data[key][frame_idx] = ball
        return data

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

    def interpolateBallTrajectory(self, data):
        """ """

    def detectHits(self, data):
        """ """

    def detectNetHits(self, data):
        """ """

    def detectArcsInterpolated(self, data):
        """ """

    def detectMissingEvents(self, data):
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
