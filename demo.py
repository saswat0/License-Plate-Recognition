from character_segmentation import segment_characters
from plate_detection import find_contours
from license_plate_extraction import extract_plate
# from CNN_model import CNN_Model
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt

original_image = cv2.imread('test.jpg')
plate_img, plate = extract_plate(original_image)
dimensions, img_dilate = segment_characters(plate)
char_list = find_contours(dimensions, img_dilate)

# for i, ch in enumerate(char_list):
#     img_ = cv2.resize(ch, (28,28))
#     cv2.imshow('Test', img_)
#     cv2.waitKey(0)

model = keras.models.load_model('model.h5')
def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char_list): #iterating over the characters
        img_ = cv2.resize(ch, (28,28))
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        y_ = model.predict_classes(img)[0] #predicting the class
        character = dic[y_] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

print(show_results())