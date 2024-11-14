import torch
import cv2
import numpy as np
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt

# Model
model = torch.hub.load('.\yolov5', 'custom',
                       source='local', path='best.pt')

# Images
img = '.\h4.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# vid_path = ""

frame = cv2.imread(img)
frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

results = model(frame2)
details = results.pandas().xyxy[0]
for index, row in details.iterrows():
    if float(row['confidence']) > 0.7:
        xm, ym, xM, yM = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        results.show()
xm = int(xm)
ym = int(ym)
xM = int(xM)
yM = int(yM)
cropped_image = frame[ym-5:yM+5, xm-5:xM+5]
cv2.imwrite("mf.png",cropped_image)



while True:

    cv2.imshow('car', frame2)
    cv2.waitKey(0)

# cap = cv2.VideoCapture(vid_path)
# ret, frame = cap.read()
# results = model(cap)
# results.show()
# cv2.imshow("vid_out", frame)


