import cv2
from cvzone.HandTrackingModule import HandDetector
from directkeys import PressKey, ReleaseKey
from directkeys import space_pressed
import time

detector=HandDetector(detectionCon=0.8, maxHands=1)

space_key_pressed=space_pressed

time.sleep(2.0)

current_key_pressed = set()

video=cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    frame = cv2.resize(frame,(780,480))
    keyPressed = False
    spacePressed=False
    key_count=0
    key_pressed=0   
    hands,img=detector.findHands(frame)
    cv2.rectangle(img, (0, 480), (300, 425),(50, 50, 255), -2)
    cv2.rectangle(img, (640, 480), (400, 425),(50, 50, 255), -2)
    if hands:
        lmList=hands[0]
        fingerUp=detector.fingersUp(lmList)
        # print(fingerUp)
        if fingerUp==[0,0,0,0,0]:
            cv2.putText(frame, 'Finger Count: 0', (20,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Jumping', (440,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            PressKey(space_key_pressed)
            spacePressed=True
            current_key_pressed.add(space_key_pressed)
            key_pressed=space_key_pressed
            keyPressed = True
            key_count=key_count+1
        if fingerUp==[0,1,0,0,0]:
            cv2.putText(frame, 'Finger Count: 1', (20,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Not Jumping', (420,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        if fingerUp==[0,1,1,0,0]:
            cv2.putText(frame, 'Finger Count: 2', (20,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Not Jumping', (420,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        if fingerUp==[0,1,1,1,0]:
            cv2.putText(frame, 'Finger Count: 3', (20,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Not Jumping', (420,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        if fingerUp==[0,1,1,1,1]:
            cv2.putText(frame, 'Finger Count: 4', (20,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Not Jumping', (420,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        if fingerUp==[1,1,1,1,1]:
            cv2.putText(frame, 'Finger Count: 5', (20,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Not Jumping', (420,460), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        if not keyPressed and len(current_key_pressed) != 0:
            for key in current_key_pressed:
                ReleaseKey(key)
            current_key_pressed = set()
        elif key_count==1 and len(current_key_pressed)==2:
            for key in current_key_pressed:
                if key_pressed!=key:
                    ReleaseKey(key)
            current_key_pressed = set()
            for key in current_key_pressed:
                ReleaseKey(key)
            current_key_pressed = set()
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
