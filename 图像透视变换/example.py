import cv2
import numpy as np

img = cv2.imread('img/test1.png')

# 原图中卡片的四个角点 顺序 左上 右上 左下 右下
pt1 = np.float32([[125, 628], [740, 293], [342, 1029], [961, 677]])

# 宽 左上角 到右上角的距离
w = np.linalg.norm(pt1[0] - pt1[1]).astype(int)
# 高 左上角 到 左下角的距离
h = np.linalg.norm(pt1[0] - pt1[2]).astype(int)
# 变换后四个点对应的位置
pt2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
# 计算变换矩阵
M = cv2.getPerspectiveTransform(pt1, pt2)
# 变换
dst = cv2.warpPerspective(img, M, (w, h))
cv2.imshow('dst', dst)
cv2.imshow('img', img)
cv2.waitKey(0)
