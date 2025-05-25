from io import BytesIO
import numpy as np
from flask import Flask, request, redirect, jsonify
from ultralytics import YOLO
import base64
import cv2
app = Flask(__name__)
model = YOLO('best.pt')


def base64_to_opencv(base64_str):
    # 移除可能存在的前缀（如 "data:image/png;base64,"）
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]

    # 解码 Base64 数据为字节流
    img_bytes = base64.b64decode(base64_str)

    # 将字节流转换为 numpy 数组
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)

    # 使用 OpenCV 解码为图像（BGR 格式）
    img_opencv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    return img_opencv


@app.route('/')
def index():
    return redirect('/static/index.html')

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()
    img = base64_to_opencv(data['image'])
    results = model.predict(source=img)
    probs = results[0].probs

    res = probs.top1
    conf = float(probs.top1conf)
    return jsonify({
        'result': res,
        'probability': f'{conf:.2f}'
    })

if __name__ == '__main__':
    app.run()