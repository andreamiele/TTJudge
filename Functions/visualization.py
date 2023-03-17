#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: visualization.py
# Created: Sunday, 12th March 2023 10:14:22 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# GitHub: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 17th March 2023 1:00:28 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023 Your Company
#
#  ==============================================================================

import cv2
import sys
import numpy as np
from os.path import abspath, dirname
from PIL import ImageFont, Image, ImageDraw

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def colorizeImg(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    return img


def drawContours(img, contours, color):
    img = img_to_color(img)
    img = cv2.drawContours(img, contours, -1, color, 3)
    return img


def contourCenter(contours):
    min_x, min_y, max_x, max_y = contourMaxMin(contours)
    return min_x + ((max_x - min_x) / 2), min_y + ((max_y - min_y) / 2)


def contourMaxMin(contours):
    min_x, min_y, max_x, max_y = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )
    for contour in contours:
        min_x = min(min_x, contour[:, :, 0].min())
        min_y = min(min_y, contour[:, :, 1].min())
        max_x = max(max_x, contour[:, :, 0].max())
        max_y = max(max_y, contour[:, :, 1].max())
    return min_x, min_y, max_x, max_y


def wContour(self, contour):
    """
    detecting if the contour is white (showing movement) or black (gaps between movement)
    """
    return True


def showTable(img, table, color=(100, 100, 100), thickness=2):
    img = img_to_color(img)
    p1, p2, p3, p4 = (
        (table[1], table[0]),
        (table[3], table[2]),
        (table[5], table[4]),
        (table[7], table[6]),
    )
    img = cv2.line(img, p1, p2, color, thickness)
    img = cv2.line(img, p2, p3, color, thickness)
    img = cv2.line(img, p3, p4, color, thickness)
    img = cv2.line(img, p4, p1, color, thickness)
    return img


def show_contour_middle_borders(img, corners):
    pass


def showEventBox(img, event):
    img = img_to_color(img)
    color = (
        (0, 255, 0)
        if event == "Bounce"
        else (255, 0, 0)
        if event == "Hit"
        else (0, 0, 255)
    )
    img = cv2.rectangle(img, (10, 10), (1910, 1070), color, 5)
    return img


def showArcD(img, data, frame_idx, arc_name, centers_name):
    img = img_to_color(img)
    for arc in data[arc_name]:
        if arc[0] <= frame_idx <= arc[1]:
            for j in range(arc[0], arc[1] + 1):
                if j in data[centers_name]:
                    c_x, c_y = data[centers_name][j]
                    img = cv2.circle(img, (int(c_x), int(c_y)), 3, (0, 0, 255), -1)
    return img


def showArcDC(img, output, i, arc_type="Interpolated Arcs"):
    img = img_to_color(img)
    for arc in output[arc_type]:
        if arc[0] <= i <= arc[1]:
            for j in range(arc[0], arc[1] + 1):
                if (
                    j in output["Final Ball Center"]
                    or j in output["Interpolated Event Center"]
                ):
                    center = (
                        output["Final Ball Center"][j]
                        if j in output["Final Ball Center"]
                        else output["Interpolated Event Center"][j]
                    )
                    img = cv2.circle(
                        img, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1
                    )
    return img


def showArcL(img, output, i, arc_type="Raw Arcs"):
    img = img_to_color(img)
    for arc in output[arc_type]:
        if arc[0] <= i <= arc[1]:
            x = []
            y = []
            for j in range(arc[0], arc[1]):
                if j in output["Phase 2 - Ball - Cleaned Contours"]:
                    c_x, c_y = contour_l_center(
                        output["Phase 2 - Ball - Cleaned Contours"][j]
                    )
                    # c_x, c_y = j
                    x.append(c_x)
                    y.append(c_y)

            model = np.poly1d(np.polyfit(x, y, 2))
            plot_x = np.linspace(min(x) - 200, max(x) + 200, 200)
            plot_y = model(plot_x)
            pts = np.array([[x, y] for x, y in zip(plot_x, plot_y)], dtype=int)
            pts = pts.reshape((-1, 1, 2))
            img = cv2.polylines(img, [pts], False, (0, 255, 0), 2)
    return img


def showExtrArcC(img, output, i):
    arc_dicts = output["Extrapolated Arc Centers"]
    for arc_dict in arc_dicts:
        arc_idxs = sorted(list(arc_dict.keys()))
        if arc_idxs[0] <= i <= arc_idxs[-1]:
            for arc_idx in arc_idxs:
                img = cv2.circle(
                    img,
                    (int(arc_dict[arc_idx][0]), int(arc_dict[arc_idx][1])),
                    3,
                    (255, 0, 0),
                    -1,
                )
    return img


def showBallBorder(img, corners):
    img = img_to_color(img)
    for c in corners:
        img = cv2.circle(img, (c[0][0], c[0][1]), 2, (0, 0, 255), -1)  # Borders in red
    return img


def showBallCenter(img, center, color=(255, 0, 0)):
    img = img_to_color(img)
    c_x, c_y = center
    img = cv2.circle(img, (int(c_x), int(c_y)), 3, color, -1)
    return img


def showFrameNb(img, frameNb):
    img = Image.fromarray(img_to_color(img))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(ROOT_PATH + "/Games/score_font.ttf", 20)
    draw.text((10, 10), str(frameNb), (255, 255, 255), font=font)
    return np.array(img)
