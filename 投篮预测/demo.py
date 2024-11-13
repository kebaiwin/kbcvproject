import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math


cap = cv2.VideoCapture('Videos/ball5.mp4')
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 124, 'vmin': 13, 'hmax': 24, 'smax': 255, 'vmax': 255}

posListX = []
posListY = []

listX = [item for item in range(0, 1300)]
start = True
prediction = False
while True:

    if start:
        if len(posListX) == 10: start = False
        success, img = cap.read()
        if not success:
            break

        img = img[0:900, :]
        imgPrediction = img.copy()
        imgResult = img.copy()
        imgBall, mask = myColorFinder.update(img, hsvVals)
        imgCon, contours = cvzone.findContours(img, mask, 200)
        if contours:
            posListX.append(contours[0]['center'][0])
            posListY.append(contours[0]['center'][1])

        if posListX:
            if len(posListX) < 18:
                coff = np.polyfit(posListX, posListY, 2)
            for i, (posX, posY) in enumerate(zip(posListX, posListY)):
                pos = (posX, posY)
                cv2.circle(imgCon, pos, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(imgResult, pos, 10, (0, 255, 0), cv2.FILLED)

                if i == 0:
                    cv2.line(imgCon, pos, pos, (0, 255, 0), 2)
                    cv2.line(imgResult, pos, pos, (0, 255, 0), 2)
                else:
                    cv2.line(imgCon, (posListX[i - 1], posListY[i - 1]), pos, (0, 255, 0), 2)
                    cv2.line(imgResult, (posListX[i - 1], posListY[i - 1]), pos, (0, 255, 0), 2)

            for x in listX:
                y = int(coff[0] * x ** 2 + coff[1] * x + coff[2])
                cv2.circle(imgPrediction, (x, y), 2, (255, 0, 255), cv2.FILLED)
                cv2.circle(imgResult, (x, y), 2, (255, 0, 255), cv2.FILLED)

            # Predict
            if len(posListX) < 10:
                a, b, c = coff
                c = c - 593
                x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))
                prediction = 300 < x < 430
            if prediction:
                cvzone.putTextRect(imgResult, "Basket", (50, 150), colorR=(0, 200, 0),
                                   scale=5, thickness=10, offset=20)
            else:
                cvzone.putTextRect(imgResult, "No Basket", (50, 150), colorR=(0, 0, 200),
                                   scale=5, thickness=10, offset=20)

        cv2.line(imgCon, (330, 593), (430, 593), (255, 0, 255), 10)
        imgResult = cv2.resize(imgResult, (0, 0), None, 0.7, 0.7)


        cv2.imshow("imgCon", imgResult)


    key = cv2.waitKey(50)
    if key == ord("s"):
        start = True