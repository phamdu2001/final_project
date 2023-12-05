import os
import cv2
from roboflow import Roboflow

def crop_main_image():
    rf = Roboflow(api_key="nXXnBLAq3cmbAyivfoyr")
    project = rf.workspace().project("detection-solqc")
    model = project.version(1).model
    folder_images_path = "D:/FinalProject/detection.v1i.yolov5pytorch/total/images/"
    file_names = os.listdir(folder_images_path)
    for file_name in file_names:
        line = model.predict("file_name", confidence=40, overlap=30).json()
        numbers = line[0].split()
        numbers = [float(x) for x in numbers]

        file_name = file_name[:-3]
        image = cv2.imread(folder_images_path + file_name + 'jpg')
        x = numbers[1] * image.shape[1]
        y = numbers[2] * image.shape[0]
        width = numbers[3] * image.shape[1]
        height = numbers[4] * image.shape[0]

        cropped_image = image[int(y - height / 2):int(y + height / 2), int(x - width / 2):int(x + width / 2)]
        name = file_name.split("_")
        resized_image = cv2.resize(cropped_image, (640,640))
        cv2.imwrite('crop/'+name[0] + ".JPG", resized_image)
        return resized_image



