# aruco 标记图片创建
import cv2
import numpy as np
img = np.zeros((500,500,3), dtype='uint8')
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
aruco_img = cv2.aruco.generateImageMarker(aruco_dict, 50, 50, img, 1)
cv2.imwrite('aruco.jpg',aruco_img)