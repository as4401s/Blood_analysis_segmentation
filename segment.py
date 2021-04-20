# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:30:33 2021

@author: asarkar
"""
import tensorflow as tf
import os
import cv2 as cv
import numpy as np
import time
import pandas as pd

#path where the original images are stored
image_path = 'D:/Blood_3/Images/'
train = os.listdir(image_path)

#path with the masks
ground_path = 'D:/Blood_3/Masks/'
train_gt = os.listdir(ground_path)

#path with csv file containing image names
df = pd.read_csv('D:/Blood_3/images.csv',names=['Name','type'],sep='.')

iou = 0
dice = 0
count = 0

#Load pretrained tensorflow model
TF_LITE_MODEL_FILE_NAME = "lite_model.tflite"
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.resize_tensor_input(input_details[0]['index'],(1,224,224,3))
interpreter.resize_tensor_input(output_details[0]['index'],(1,224,224,1))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])

image_size=224

#crop image
def crop(img):
    crop_img = img[0:200,200:700]
    return crop_img

#loop over each image
for i, item in df.iterrows():
    
    #read original image
    img = cv.imread(image_path + item[0]+'.png')
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    img = img/255.0
    display = img
    
    image = crop(img)
    image = cv.resize(image,(image_size,image_size))
    image = np.array(image,dtype=np.float32)
    
    #read image mask    
    mask = cv.imread(ground_path + item[0]+'_0_region.png')
    mask = cv.cvtColor(mask,cv.COLOR_BGR2RGB)
    original_mask = mask
    mask = mask/255.0
    
    #count number of white pixels in mask
    retval_1=cv.countNonZero(mask[:,:,0])
    
    #predict the mask from the deep learning model
    test_image =np.expand_dims(image, axis=0)
    
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    
    
    result= interpreter.get_tensor(output_details[0]['index'])              # used to predict the result
    result=result>0.5
    predict_result =np.squeeze(result,axis=0)
    stop = time.perf_counter()
    elapsed_time = format((stop-start)*1000,'.3f')
    print('Time in ms = ',elapsed_time)
    
    #resize prediction to original image size
    a= np.zeros((224,1280,1),dtype=np.uint8)
    m = predict_result.astype(np.uint8)
    dim = (500,200)
    resized = cv.resize(m,dim,interpolation = cv.INTER_AREA)
    resized = np.expand_dims(resized,axis= -1)
    a [0:200,200:700,:] = resized
    segment = a*255
    
    count = count +1
    
    #count number of white pixels in predicted mask
    retval2=cv.countNonZero(segment)
    
    #create 3 channel image from binary image for concatenation
    segment2 = cv.merge((segment,segment,segment))
    
    #convert uint8 binary mask to float64 for blensing with original image
    segment3 = np.array(segment2, dtype=np.float64)
    
    defect_map = segment3[:, :, 1] > 0.5
    label_image = original_mask[:,:,0] > 0.5
    IoU = (defect_map * label_image).sum() / (defect_map + label_image).sum()
    print ('IOU % = ',np.round(IoU*100,2))
    
    iou = iou+IoU
    
    intersection = np.logical_and(defect_map, label_image)
    Dice = 2. * intersection.sum() / (defect_map.sum() + label_image.sum())
    print ('Dice Coefficient % = ',np.round(Dice*100,2))
    
    dice = dice + Dice
    
    #blending predicted mask with original image
    dst = cv.addWeighted(segment3, 1, img, 1, 0.0)
    
    #font type for text
    font                   = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,200)
    fontScale              = 0.7
    fontColor              = (255,255,255)
    lineType               = 2

    #add text to output image
    cv.putText(img,'Original Image', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv.putText(mask,f'Original Mask: {retval_1} Pixels', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv.putText(segment2,f'Predicted Mask: {retval2} Pixels', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv.putText(dst,'Merged Image with Predicted Mask', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    
    #concatenate and display all images together
    img_concate_Verti=np.concatenate((img,mask,segment2,dst),axis=0)
    
    #display the images
    
    cv.imshow('concatenated_Vertical',img_concate_Verti)
    
    
    #press q to quit
    if cv.waitKey(0) == ord('q'):
        break
    
cv.destroyAllWindows()

iou = (iou/count)*100
iou = np.round(iou,2)

dice = (dice/count)*100
dice = np.round(dice,2)

print ('Total IOU = ',iou)
print ('Total dice = ', dice)