from character_segmentation import segment_characters
from plate_detection import find_contours
from license_plate_extraction import extract_plate
from results import show_results

import cv2
import numpy as np
import keras

original_image = cv2.imread('test.jpg')
plate_img, plate = extract_plate(original_image)
dimensions, img_dilate = segment_characters(plate)
char_list = find_contours(dimensions, img_dilate)

model = keras.models.load_model('model.h5')

print(show_results(model, char_list))