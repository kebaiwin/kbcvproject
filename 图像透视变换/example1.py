# -*- coding: utf-8 -*-
# 通过轮廓检测 确定坐标
import cv2
import numpy as np

img = cv2.imread('img/test1.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh =cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 4:
        left_top = approx[1][0]
        right_top = approx[0][0]
        left_bottom = approx[2][0]
        right_bottom = approx[3][0]
        pt1 = np.float32([left_top, right_top, left_bottom, right_bottom])
        w = np.linalg.norm(left_top-right_top).astype(int)
        h = np.linalg.norm(left_top-left_bottom).astype(int)
        pt2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(pt1, pt2)
        dst = cv2.warpPerspective(img, M, (w,h))
        cv2.imshow('dst', dst)
        cv2.circle(img, tuple(approx[0][0]), 10, (0, 0, 255), -1)
        cv2.circle(img, tuple(approx[1][0]), 10, (255, 0, 0), -1)
        cv2.circle(img, tuple(approx[2][0]), 10, (0, 255, 0), -1)
        cv2.circle(img, tuple(approx[3][0]), 10, (0, 0, 0), -1)
cv2.imshow('thresh', thresh)

cv2.imshow('image', img)
cv2.waitKey()