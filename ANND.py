#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:15:04 2019

@author: 2020shatgiskessell
"""
import cv2
import numpy as np
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


#from quickdraw import QuickDrawData

doodle = cv2.imread("/Users/2020shatgiskessell/Desktop/ANND/test6.png")

start = timeit.default_timer()

def isolate_doodle(doodle):
    doodles = []
    xs = []
    ys = []
    imgray = cv2.cvtColor(doodle, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 100,255, cv2.THRESH_BINARY)
#    kernel = np.ones((3,1),np.uint8)
#    doodle = cv2.erode(thresh, kernel, 1)
#    doodle = cv2.dilate(doodle,kernel,1 )
    doodle = thresh
    contours, hierarchy = cv2.findContours(doodle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    try:
        contours.remove(c)
    except ValueError:
        pass
    doodle = cv2.cvtColor(doodle, cv2.COLOR_GRAY2BGR)
    validated_contours = []
    for i in range(len(contours)):
        contour = contours[i]
        x,y,w,h = cv2.boundingRect(contour)
        xs.append(x)
        ys.append(y)
        area = cv2.contourArea(contour)
        if area >800: 
        #and hierarchy[0,i,3]==-1:
            doodle_img = doodle[y-5: y+h+5, x-5: x+w+5]
            #doodle = cv2.rectangle(doodle,(x-5,y-5),(x+w+5,y+h+5),(0,255,0),2)
            #cv2.imshow("doodle img", doodle_img)
            #cv2.waitKey(0)
            doodles.append(doodle_img)
    return doodle, doodles, xs, ys

def array_change(arr3):
    arr = []
    for i in range(len(arr3)):
        for j in range(len(arr3[0])):
            if arr3[i][j][0] == 255:
                arr3[i][j] = 0
            else:
                arr3[i][j] = 1
            arr.append(arr3[i][j])
    return arr

def main(img):
    identified_images = []
    h,w,c = img.shape
    blank_image = cv2.imread("/Users/2020shatgiskessell/Desktop/ANND/blank.png")
    label_dict = {0: 'sun', 1: 'castle', 2: 'cake', 3:'dog', 4: 'face', 5: 'computer', 6: 'house', 7: 'airplane', 8:'cloud', 9: 'apple'}
    doodle_images = []
    doodle_images_h = []
    doodle_images_w = []
    image, doodles,xs,ys = isolate_doodle(img)
    for i in range (len(doodles)):
        doodle = doodles[i]
        h,w, c = doodle.shape
        doodle = cv2.resize(doodle, (28,28))
        doodle = doodle.tolist()
    #            
        doodle = array_change(doodle)
        doodle = np.array(doodle)
        doodle_images_h.append(h)
        doodle_images_w.append(w)
        doodle_images.append(doodle)
        
    loaded_model = joblib.load('finalized_model_cnn.sav')
    #result = loaded_model.predict(doodle_images)
    result = loaded_model.predict(doodle_images)
    for i in range(len(result)):
        identified = str(label_dict.get(int(result[i])))
        #pred = quick_draw (doodle_images)
        print("prediction is: " + identified)
        identified_image = cv2.imread("/Users/2020shatgiskessell/Desktop/ANND/" + identified + ".png")
        identified_image = cv2.resize(identified_image, (doodle_images_h[i],doodle_images_w[i]))
        offset = 0
        blank_image[ys[i]:ys[i]+identified_image.shape[0] + offset, xs[i]:xs[i]+identified_image.shape[1] + offset] = identified_image
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    cv2.imshow("original", img)
    cv2.imshow("blank_image", blank_image)
    cv2.waitKey(0)
#quick_draw2 ()
main(doodle)

cv2.imshow('doodleD', image)
#cv2.imshow('isolated doodle', isolated_doodle)

cv2.waitKey(0)
cv2.destroyAllWindows()