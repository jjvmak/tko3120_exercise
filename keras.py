# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 08:32:56 2019

@author: makin
"""

import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize, rescale, rotate, setup, warp
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import rgb2gray
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def create_image_array(location):
    """Returns array of images of given locations."""
    images = []
    urls = np.loadtxt(location, dtype='U100')
    print('reading urls from ' + location)
    print(repr(len(urls)) + ' urls found')
    print()
    # for dev decrease the range! TODO: change back to len(array)
    for i in range(len(urls)):
        print('reading image ' + repr(i) + ' from ' + urls[i])
        try:
            image = io.imread(urls[i])
            images.append(rgb2gray(image))
            images.append(rgb2gray(rotate(image, angle=np.pi/2)))
        except:
            print("An exception occurred")
    return images


def resize_images(image_array, x, y):
    """Returns array of resized images."""
    resized_images = []
    for i in range(len(image_array)):
        img = image_array[i]
        # TODO: what are right configs?
        resized = resize(img, (x, y), anti_aliasing='reflect', mode='reflect')
        resized_images.append(resized)
    return resized_images

def make_y_for_x(data, label):
    y = []
    for i in range(len(data)):
        y.append(label)
    return y


def build_model():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape = (28 * 28,)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
    model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

print()

x = 28
y = 28
honeycomb_resized = resize_images(create_image_array('honeycomb.txt'), x, y)
birdnest_resized = resize_images(create_image_array('birdnests.txt'), x, y)
lighthouse_resized = resize_images(create_image_array('lighthouse.txt'), x, y)

honeycomb_x = tf.keras.utils.normalize(honeycomb_resized, axis=1)
birdnest_x = tf.keras.utils.normalize(birdnest_resized, axis=1)
lighthouse_x = tf.keras.utils.normalize(lighthouse_resized, axis=1)

honeycomb_y = make_y_for_x(honeycomb_x, 0)
birdnest_y = make_y_for_x(birdnest_x, 1)
lighthouse_y = make_y_for_x(lighthouse_x, 2)

data_x = np.concatenate((honeycomb_x, birdnest_x, lighthouse_x), axis=0)
data_y = np.concatenate((honeycomb_y, birdnest_y, lighthouse_y), axis=0)

data_x = np.array(data_x).reshape(len(data_x), x*y)

data_y_enc = tf.keras.utils.to_categorical(data_y)

data = np.concatenate((data_x, data_y_enc), axis=1)
np.random.shuffle(data)

col = data.shape[1]

data[:,[col-3, col-2, col-1]]
data[:,:col-3]

model = build_model()
model.fit(data[:,:col-3], data[:,[col-3, col-2, col-1]], epochs=5)


