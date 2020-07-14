import cv2
import numpy as np

def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results(model, char_list):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char_list): #iterating over the characters
        img_ = cv2.resize(ch, (28,28))
        # cv2.imshow('Test', img_)
        # cv2.waitKey(0)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        # print(np.argmax(model.predict(img), axis=-1))
        # y_ = model.predict_classes(img)[0] #predicting the class
        # y_ = np.argmax(model.predict_classes(img), axis=-1)
        y_ = np.argmax(model.predict(img), axis=-1)[0]
        character = dic[y_] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number