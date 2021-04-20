# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:30:33 2021

@author: asarkar
"""

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf

image_path = 'D:/Blood_1/Classify_Images/'

name_path = 'D:/Blood_1/Classify_Results.csv'

Labels = ['Blue','White']

df = pd.read_csv(name_path)

model_path = 'D:/Blood_1/model.tflite'

interpreter = tf.lite.Interpreter(model_path = model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

def display_image(image,original_label,predicted_label, accuracy):
    
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,200)
    fontScale              = 0.7
    fontColor              = (255,255,255)
    lineType               = 2

    cv.putText(image,f'Original Label = {original_label}, Predicted Label = {predicted_label}, Accuracy = {accuracy} %', 
               bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    
    cv.imshow('Image',image)
    
image_size=224

for i,item in df.iterrows():
    
    image = cv.imread(image_path + item[0] + '.png')
    image_1 = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    display = image
    image = image_1/255.0
    image = cv.resize(image,(image_size,image_size))
    image = np.expand_dims(image, axis=0)
    
    image = np.array(image,dtype=np.float32)
    
    original_label = item[1]
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
    prediction_classes = np.where(tflite_model_predictions<0.5, 0,1)
    accuracy = float(tflite_model_predictions)
    
    if prediction_classes == 0:
        display_image(display,original_label,Labels[0],format((1-accuracy)*100, '.2f'))
        
    else:
        display_image(display,original_label,Labels[1],format((accuracy*100), '.2f'))
        
    if cv.waitKey(0) == ord('q'):
        break
        
cv.destroyAllWindows()