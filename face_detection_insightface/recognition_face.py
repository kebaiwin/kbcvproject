import cv2
import numpy as np
import insightface
from numpy.linalg import norm

detector = insightface.app.FaceAnalysis(allowed_modules=['recognition','detection'])
detector.prepare(ctx_id=0,det_size=(640,640))

musk_img = cv2.imread('img/musk_base.jpg')
musk_face = detector.get(musk_img)
musk_embedding = musk_face[0].normed_embedding

img = cv2.imread('img/musk.jpg')
faces = detector.get(img)
for face in faces:
    bbox = face.bbox.astype(np.int32)
    embedding = face.normed_embedding
    sim = np.dot(embedding,musk_embedding)/(norm(musk_embedding)*norm(embedding))
    score = (sim*100).astype(np.int32)
    print('sim:',sim)
    if score >50:
        cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
        cv2.putText(img, f'musk,score{score}', (bbox[0],bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(img,(bbox[0],bbox[3]),(bbox[2],bbox[3]), (0,255,0), 2)
    else:
        cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,255), 2)
        cv2.putText(img, 'unknown', (bbox[0],bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
cv2.imshow('img',img)
cv2.waitKey(0)


