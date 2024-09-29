# 根据鼠标选择四个角点，并计算透视变换
import cv2
import numpy as np

coordinates = []
# 鼠标回调函数
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在图像上绘制点
        cv2.circle(image, (x, y), 15, (0, 255, 0), -1)  # 绘制绿色圆点
        # 显示坐标文本
        cv2.putText(image, f"({x}, {y})", (x-100, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        coordinates.append(np.array([x, y]))

        if len(coordinates) == 4:
            w = np.linalg.norm(coordinates[0] - coordinates[1]).astype(np.int32)
            h = np.linalg.norm(coordinates[0] - coordinates[2]).astype(np.int32)
            print(w,h)
            pt1 = np.float32([coordinates])
            pt2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            M = cv2.getPerspectiveTransform(pt1, pt2)
            warped = cv2.warpPerspective(img, M, (w, h))
            cv2.imshow("Warped Image", warped)
        cv2.imshow("Image", image)

# 读取图像
img = cv2.imread('img/test2.jpg')
image = img.copy()
# 显示原图
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

# 等待按键
cv2.waitKey(0)
cv2.destroyAllWindows()
