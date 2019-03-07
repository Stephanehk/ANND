#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:06:24 2019

@author: 2020shatgiskessell
"""
import cv2
import numpy as np
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')


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
    model.add(Dense(output_dim = num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    
    label_dict = {0: 'sun', 1: 'castle', 2: 'cake', 3:'dog', 4: 'face', 5: 'computer', 6: 'house', 7: 'airplane', 8:'cloud', 9: 'apple'}

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
    y_pred_cnn = model.predict_classes(X_test_cnn, batch_size=32, verbose=0)
    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    print ('CNN accuracy: ',acc_cnn)
    filename = 'finalized_model_cnn.sav'
    joblib.dump(model, filename)
cnn()