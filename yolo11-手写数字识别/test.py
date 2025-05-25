from ultralytics import YOLO

model = YOLO('best.pt')
for i in range(10):
    imgPath = f'input/{i}.png'
    results = model.predict(source=imgPath)
    probs = results[0].probs
    conf = float(probs.top1conf)
    predict = probs.top1
    print(f'真实值： {i}, 预测值： {predict:}, 分数：{conf:.2f}')


