import library
import os
import cv2
import numpy as np
import pickle as pkl

X = []
y = []
if __name__ == '__main__':
    path = "D:/FinalProject/raw_data/"
    file_names = os.listdir(path + 'labels/')
    for file_name in file_names:
        image = library.crop_image(path, file_name)
        name = file_name.split("_")
        image_segment = library.segment(image, name[0])
        NDVI_segment = library.RGN2NVDI_segment(image_segment, name[0])
        hist, bins = np.histogram(NDVI_segment.flatten(), bins=256, range=[20, 240])
        norm_hist = hist / sum(hist)
        X.append(norm_hist)
        if(file_name[0] == 'g'): y.append(0)
        elif(file_name[0] == 'r'): y.append(1)
        else: y.append(2)
    
    with open("train.pkl", "wb") as f:
        pkl.dump([X, y], f)