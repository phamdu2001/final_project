import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

folder_path = "D:/FinalProject/segment/"
folder_write_path = "D:/FinalProject/histogram/"

# def RGN2NVDI(image):
#     red_band = image[:, :, 0].astype(float)
#     nir_band = image[:, :, 2].astype(float)
#     ndvi = (nir_band - red_band) / (nir_band + red_band)
#     ndvi_normalized = (ndvi + 1) * 127.5
#     return ndvi_normalized.astype(np.uint8)

def histogram(image, file_name):
    # Tính histogram của ảnh
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Hiển thị histogram
    plt.figure(figsize=(10, 5))
    plt.hist(image.flatten(), bins=256, range=[20, 230], color='gray')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of the Image')
    plt.savefig(folder_write_path + file_name)

file_names = os.listdir(folder_path)
for file_name in file_names:
    image = cv2.imread(folder_path + file_name)  
    # NVDI = RGN2NVDI(image)
    histogram(image[:,:,2], file_name)
    