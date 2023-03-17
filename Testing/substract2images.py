#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: substract2images.py
# Created: Friday, 17th March 2023 11:22:58 am
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# Github: https://www.github.com/andreamiele
# -----
# Last Modified: Friday, 17th March 2023 11:23:48 am
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
#
# -----
# Copyright (c) 2023
#
#  ==============================================================================

import cv2

circle = cv2.imread("circle.png")
star = cv2.imread("star.png")
subtracted = cv2.subtract(star, circle)
cv2.imshow("image", subtracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)


def frameDifference(pframe, frame):
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


test = frameDifference(circle, star)
cv2.imshow("image", test)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
