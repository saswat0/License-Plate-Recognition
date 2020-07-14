from character_segmentation import segment_characters
from plate_detection import find_contours
from license_plate_extraction import extract_plate
from results import show_results

import cv2
import numpy as np
import keras

original_image = cv2.imread('test.jpeg')
plate_img, plate = extract_plate(original_image)
# cv2.imshow('Test', plate_img)
# cv2.waitKey(0)
dimensions, img_dilate = segment_characters(plate)
cv2.imshow('Original Image', original_image)
cv2.imshow('Detected Plate on image', plate_img)
cv2.imshow('Extracted Plate', plate)
cv2.imshow('Dilated Image', img_dilate)
cv2.waitKey(0)
char_list = find_contours(dimensions, img_dilate)

model = keras.models.load_model('model.h5')

print(show_results(model, char_list))