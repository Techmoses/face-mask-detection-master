# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:19:30 2020

@author: Jinu
"""

import os
import shutil
import cv2


#declared global path by nasa

path ="D:/online chapters/anaconda opencv/ml-part/dataset/without_mask/"  #input path
destination = "D:/online chapters/anaconda opencv/ml-part/dataset/without_mask2/"  #output path

#IMAGE_DIMS = (256, 256)

def rename(): 
    i=0
    for filename in os.listdir(path):
        print(path+filename) 
        dest = "image_"+str(i)+".jpg"
        print("destination : ",destination)
        os.rename(path+filename, destination+dest)
        print("i: ",i)
        i=i+1


rename()
print("done")