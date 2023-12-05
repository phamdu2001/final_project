import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl

X = []
y = []
img = cv2.imread('D:/FinalProject/NDVI/green93.jpg',0)
hist, bins = np.histogram(img.flatten(), bins=256, range=[20, 240])
print(hist/sum(hist))
X.append(hist/sum(hist))
y.append(1)

with open("train.pkl", "wb") as f:
    pkl.dump([X, y], f)

#to load it
with open("train.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)

print(x_train)
print(y_train)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()