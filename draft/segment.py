import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def RGN2NVDI(image):
    red_band = image[:, :, 0].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_normalized = (ndvi + 1) * 127.5
    return ndvi_normalized.astype(np.uint8)

def RGN2NVDI_green(image):
    red_band = image[:, :, 1].astype(float)
    nir_band = image[:, :, 2].astype(float)
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi_normalized = (ndvi + 1) * 127.5
    return ndvi_normalized.astype(np.uint8)

folder_path = "D:/FinalProject/crop/"
folder_write_path = "D:/FinalProject/segment/"
file_names = os.listdir(folder_path)
for file_name in file_names:
    image = cv2.imread(folder_path + file_name)  
    NVDI = RGN2NVDI(image)
    NVDI_gr = RGN2NVDI_green(image)

    _, segmented_image = cv2.threshold(NVDI, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image = np.logical_not(segmented_image).astype(int)
    segmented_image = segmented_image.astype(np.uint8)

    _, segmented_image_gr = cv2.threshold(NVDI_gr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image_gr = np.logical_not(segmented_image_gr).astype(int)
    segmented_image_gr = segmented_image_gr.astype(np.uint8)

    mask = cv2.bitwise_and(segmented_image_gr,segmented_image_gr,mask=segmented_image)
    result = cv2.bitwise_and(image,image,mask=mask)
    cv2.imwrite('segment/' + file_name,result)