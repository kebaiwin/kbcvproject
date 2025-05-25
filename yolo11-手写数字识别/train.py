from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolo11s-cls.pt")
    # 训练模式
    results = model.train(data="mnist", epochs=50, imgsz=32)
