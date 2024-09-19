import mediapipe as mp
import cv2
import numpy as np
segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture('data/person.mp4')

bgcap = cv2.VideoCapture('data/scenery.mp4')
while True:
    ret, frame = cap.read()
    ret2, bgframe = bgcap.read()
    if not ret or not ret2:
        break
    imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmentation.process(imgrgb)
    mask = result.segmentation_mask
    mask = (mask > 0.9).astype(np.uint8)*255
    mask = cv2.medianBlur(mask, 9)
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    bgframe = cv2.bitwise_and(bgframe, bgframe, mask=~mask)

    outframe = cv2.add(bgframe,frame)

    cv2.imshow('res',outframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break