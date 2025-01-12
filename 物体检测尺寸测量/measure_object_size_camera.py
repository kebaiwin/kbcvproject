import cv2

import numpy as np

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
# Load Cap
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, img = cap.read()
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if corners:
        int_corners = np.int32(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)
        aruco_perimeter = cv2.arcLength(corners[0], True)
        # 计算图像中每厘米对应的像素值
        pixel_cm_ratio = aruco_perimeter / 20
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                objects_contours.append(cnt)

        for cnt in objects_contours:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()