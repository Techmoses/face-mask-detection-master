# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:19:30 2020

@author: Jinu
"""

import os
import shutil
import cv2


Person_model = cv2.CascadeClassifier('Person_model.xml')
#declared global path 

path ="D:/online chapters/anaconda opencv/ml-part/dataset/without_mask/"  #input path
destination = "D:/online chapters/anaconda opencv/ml-part/dataset/without_mask_crop/"  #output path

IMAGE_DIMS = (256, 256)

def rename(): 
    i=0
    for filename in os.listdir(path):
        print(filename)
        img = cv2.imread(path+filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        person = Person_model.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in person:
            crop = img[y:y+h+50, x:x+w+50]
        print("flename",filename)
        cv2.imwrite(destination+"image-"+str(i)+".jpg", crop) 
        print("i: ",i)
        i=i+1


rename()
print("done")