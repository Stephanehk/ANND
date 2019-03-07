#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:15:04 2019

@author: 2020shatgiskessell
"""
print ("running")
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

doodle = cv2.imread("/Users/2020shatgiskessell/Desktop/ANND/flask/images/test7.png")

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
    for i in range(len(contours)):
        contour = contours[i]
        x,y,w,h = cv2.boundingRect(contour)
        xs.append(x)
        ys.append(y)
        area = cv2.contourArea(contour)
        if area >1000 and hierarchy[0,i,2]==-1:
            doodle_img = doodle[y-2: y+h+2, x-2: x+w+2]
            #doodle = cv2.rectangle(doodle,(x-2,y-2),(x+w+2,y+h+2),(0,255,0),2)
            #cv2.imshow("doodle img", doodle_img)
            #cv2.waitKey(0)
            doodles.append(doodle_img)
            #cv2.drawContours(doodle, [contour], 0, (0,255,0),-1)
    return thresh, doodles, xs, ys


def cnn_model(num_classes):
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def cnn():
    sun = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_sun.npy')
    castle = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_castle.npy')
    cake = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_cake.npy')
    dog = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_dog.npy')
    face = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_face.npy')
    computer = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_computer.npy')
    house = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_house.npy')
    airplane = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_airplane.npy')
    cloud = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_cloud.npy')
    apple = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_apple.npy')

    sun = np.c_[sun, np.zeros(len(sun))]
    castle = np.c_[castle, np.ones(len(castle))]
    cake = np.c_[cake, 2 *np.ones(len(cake))]
    dog = np.c_[dog, 3 *np.ones(len(dog))]
    face = np.c_[face, 4 *np.ones(len(face))]
    computer = np.c_[computer, 5 *np.ones(len(computer))]
    house = np.c_[house, 6 *np.ones(len(house))]
    airplane = np.c_[airplane, 7 *np.ones(len(airplane))]
    cloud = np.c_[cloud, 8 *np.ones(len(cloud))]
    apple = np.c_[apple, 9 *np.ones(len(apple))]

    #doodles = np.c_[banana, 2*np.ones(len(doodles))]

    X = np.concatenate((sun[:5000,:-1], castle[:5000,:-1], cake[:5000,:-1], dog[:5000,:-1], face[:5000,:-1],dog[:5000,:-1],computer[:5000,:-1],house[:5000,:-1],airplane[:5000,:-1], cloud[:5000,:-1], apple[:5000,:-1]), axis=0).astype('float32') # all columns but the last
    y = np.concatenate((sun[:5000,-1], castle[:5000,-1], cake[:5000,-1], dog[:5000,-1], face[:5000,-1],dog[:5000,-1],computer[:5000,-1],house[:5000,-1],airplane[:5000,-1], cloud[:5000,-1], apple[:5000,-1]), axis=0).astype('float32') # all columns but the last
    X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.2,random_state=0)


    y_train_cnn = np_utils.to_categorical(y_train)
    y_test_cnn = np_utils.to_categorical(y_test)
    num_classes = y_test_cnn.shape[1]
    X_train_cnn = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test_cnn = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    # build the model
    model = cnn_model(num_classes)
    # Fit the model
    model.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=10, batch_size=200)
    # Final evaluation of the model
    scores = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
    print('Final CNN accuracy: ', scores[1])
    filename = 'finalized_model_cnn.sav'
    joblib.dump(model, filename)


def quick_draw2 ():
    sun = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_sun.npy')
    castle = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_castle.npy')
    cake = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_cake.npy')
    dog = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_dog.npy')
    face = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_face.npy')
    computer = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_computer.npy')
    house = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_house.npy')
    airplane = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_airplane.npy')
    cloud = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_cloud.npy')
    apple = np.load('/Users/2020shatgiskessell/Downloads/full_numpy_bitmap_apple.npy')

    sun = np.c_[sun, np.zeros(len(sun))]
    castle = np.c_[castle, np.ones(len(castle))]
    cake = np.c_[cake, 2 *np.ones(len(cake))]
    dog = np.c_[dog, 3 *np.ones(len(dog))]
    face = np.c_[face, 4 *np.ones(len(face))]
    computer = np.c_[computer, 5 *np.ones(len(computer))]
    house = np.c_[house, 6 *np.ones(len(house))]
    airplane = np.c_[airplane, 7 *np.ones(len(airplane))]
    cloud = np.c_[cloud, 8 *np.ones(len(cloud))]
    apple = np.c_[apple, 9 *np.ones(len(apple))]

    #doodles = np.c_[banana, 2*np.ones(len(doodles))]

    X = np.concatenate((sun[:5000,:-1], castle[:5000,:-1], cake[:5000,:-1], dog[:5000,:-1], face[:5000,:-1],dog[:5000,:-1],computer[:5000,:-1],house[:5000,:-1],airplane[:5000,:-1], cloud[:5000,:-1], apple[:5000,:-1]), axis=0).astype('float32') # all columns but the last
    y = np.concatenate((sun[:5000,-1], castle[:5000,-1], cake[:5000,-1], dog[:5000,-1], face[:5000,-1],dog[:5000,-1],computer[:5000,-1],house[:5000,-1],airplane[:5000,-1], cloud[:5000,-1], apple[:5000,-1]), axis=0).astype('float32') # all columns but the last
    X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.2,random_state=0)


    clf_rf = RandomForestClassifier(n_estimators=100,n_jobs=-1, random_state=0)
    clf_rf.fit(X_train, y_train)
    filename = 'finalized_model.sav'
    joblib.dump(clf_rf, filename)




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
    label_dict = {0: 'house', 1: 'castle', 2: 'cake', 3:'dog', 4: 'face', 5: 'computer', 6: 'sun', 7: 'airplane', 8:'cloud', 9: 'apple'}
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

    loaded_model = joblib.load('finalized_model.sav')
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
    cv2.imshow("blank_image", blank_image)
    cv2.imshow("original", img)
    cv2.waitKey(0)
#quick_draw2 ()
main(doodle)
stop = timeit.default_timer()
print('Time: ', stop - start)

#isolated_doodle, doodles = isolate_doodle(doodle)
#cv2.imshow('doodle', doodle)
#cv2.imshow('isolated doodle', isolated_doodle)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
