# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # for file/URI/PIL/cv2/np inputs and NMS


if __name__ == '__main__':
    #使用するカメラを指定する
    camera = cv2.VideoCapture(0)

    while True:
        #映像を取得する
        ret, frame = camera.read()
        new_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred = model(new_image)
        pred.save()

        img = Image.open('results0.jpg')
        new_image = np.array(img, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

        #取得した映像を表示する
        cv2.imshow("camera",new_image)

        key = cv2.waitKey(10)
        # Escキーが押されたら
        if key == 27:
            cv2.destroyAllWindows()
            break

        # print(results.xyxy[0])  # print img1 predictions
        boxes = pred.xyxy[0].numpy()
        for (x,y,w,h,a,b) in boxes:
            print(x,y,w,h,a,b)
