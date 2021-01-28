import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)  # for file/URI/PIL/cv2/np inputs and NMS

# Images
img = Image.open('bus.jpg')  # PIL image

# Inference
results = model(img)  # includes NMS

results.print()  # print results to screen
results.show()  # display results
results.save()  # save as results1.jpg, results2.jpg... etc.

# print(results.xyxy[0])  # print img1 predictions
boxes = results.xyxy[0].numpy()
for (x,y,w,h,a,b) in boxes:
    print(x,y,w,h,a,b)
