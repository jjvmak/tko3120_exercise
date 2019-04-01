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
from sklearn.decomposition import PCA
from scipy import stats


from skimage import img_as_ubyte as eight_bit
from skimage.feature import greycomatrix, greycoprops
import statistics


def create_image_array(location):
    """Returns array of images of given locations."""
    images = []
    urls = np.loadtxt(location, dtype='U100')
    print('reading urls from ' + location)
    print(repr(len(urls)) + ' urls found')
    print()
    # for dev decrease the range! TODO: change back to len(array)
    for i in range(4):
        print('reading image ' + repr(i) + ' from ' + urls[i])
        try:
            image = io.imread(urls[i])
            images.append(image)
        except:
            print("An exception occurred")
    return images


def resize_images(image_array, x, y):
    """Returns array of resized images."""
    resized_images = []
    resized_gray_images = []
    for i in range(len(image_array)):
        img = image_array[i]
        # TODO: what are right configs?
        resized = resize(img, (x, y), anti_aliasing='reflect', mode='reflect')
        resized_images.append(resized)
        resized_gray_images.append(rgb2gray(resized))
    return resized_images, resized_gray_images

def make_y_for_x(data, label):
    y = []
    for i in range(len(data)):
        y.append(label)
    return y


def build_model(data_shape):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_dim=data_shape))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def first_order_texture_measures(img, show_plts):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r_mean = np.mean(r, axis=1)
    g_mean = np.mean(g, axis=1)
    b_mean = np.mean(b, axis=1)

    r_var = np.var(r, axis=1)
    g_var = np.var(g, axis=1)
    b_var = np.var(b, axis=1)

    if show_plts:
        fig, ax = plt.subplots()
        ax.plot(r_mean, color='red')
        ax.plot(g_mean, color='green')
        ax.plot(b_mean, color='blue')
        ax.grid()
        plt.show()

        fig, bx = plt.subplots()
        bx.plot(r_var, color='red')
        bx.plot(g_var, color='green')
        bx.plot(b_var, color='blue')
        bx.grid()
        plt.show()
    fetures = np.concatenate(
        (r_mean, g_mean, b_mean, r_var, g_var, b_var), axis=0)

    return fetures.reshape(-1, len(fetures))


def extract_rgb_features(image_array):
    features = first_order_texture_measures(image_array[0], False)
    for i in range(len(image_array)):
        if (i != 0):
            features = np.concatenate((features, first_order_texture_measures(image_array[i], False)), axis=0)
    return features


def extract_glcm_features_for_image(image, other_image):
    # http://scikit-image.org/docs/0.9.x/auto_examples/plot_glcm.html
    # I used that tutorial with few modifications
    image = eight_bit(image)
    other_image = eight_bit(other_image)

    PATCH_SIZE = 150

    locations = [(0, 0), (0, 150), (150, 0), (150, 150)]
    patches = []

    other_patches = []

    for loc in locations:
        patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                       loc[1]:loc[1] + PATCH_SIZE])
        other_patches.append(other_image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

    contrast = []
    dissimilarity = []
    homogeneity = []
    correlation = []
    asm = []
    for i, patch in enumerate(patches + other_patches):
        glcm = greycomatrix(patch, [1], [0, np.pi / 2, np.pi], 256, symmetric=True, normed=True)
        contrast.append(greycoprops(glcm, 'contrast')[0, 0])
        dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        homogeneity.append(greycoprops(glcm, 'homogeneity')[0, 0])
        correlation.append(greycoprops(glcm, 'correlation')[0, 0])
        asm.append(greycoprops(glcm, 'ASM')[0, 0])

    return contrast, dissimilarity, homogeneity, correlation, asm


def extract_glcm_features_for_set(set, use_means):
    features = []
    for i in range(len(set)):
        contrast, dissimilarity, homogeneity, correlation, asm = \
            extract_glcm_features_for_image(set[i], ref_img)
        features_for_row = np.array([])

        if use_means:

            features_for_row = np.append(features_for_row, statistics.mean(contrast))
            features_for_row = np.append(features_for_row, statistics.mean(dissimilarity))
            features_for_row = np.append(features_for_row, statistics.mean(homogeneity))
            features_for_row = np.append(features_for_row, statistics.mean(correlation))
            features_for_row = np.append(features_for_row, statistics.mean(asm))

            features_for_row = features_for_row.reshape(-1, len(features_for_row))

        else:
            for con in range(len(contrast)):
                features_for_row = np.append(features_for_row, contrast[con])

            features_for_row = features_for_row.reshape(-1, len(features_for_row))

            for dis in range(len(dissimilarity)):
                features_for_row = np.append(features_for_row, dissimilarity[dis])

            features_for_row = features_for_row.reshape(-1, len(features_for_row))

            for hom in range(len(homogeneity)):
                features_for_row = np.append(features_for_row, homogeneity[hom])

            features_for_row = features_for_row.reshape(-1, len(features_for_row))

            for cor in range(len(correlation)):
                features_for_row = np.append(features_for_row, correlation[cor])

            features_for_row = features_for_row.reshape(-1, len(features_for_row))

            for a in range(len(asm)):
                features_for_row = np.append(features_for_row, asm[a])

            features_for_row = features_for_row.reshape(-1, len(features_for_row))

        if len(features) == 0:
            features = features_for_row
        else:
            features = np.concatenate((features, features_for_row), axis=0)

    return features

def load_ref_image():
    try:
        image = io.imread(
            'https://images.freeimages.com/images/large-previews/e71/frog-1371919.jpg')
        image_rez, gray_rez = resize_images([image], 300, 300)

    except:
        print("An exception occurred")
    return gray_rez[0]

print()

ref_img = load_ref_image()

x = 300
y = 300
honeycomb_resized, honeycomb_resized_gray = resize_images(create_image_array('honeycomb.txt'), x, y)
birdnest_resized, birdnest_resized_gray = resize_images(create_image_array('birdnests.txt'), x, y)
lighthouse_resized, lighthouse_resized_gray = resize_images(create_image_array('lighthouse.txt'), x, y)

#honeycomb_x = tf.keras.utils.normalize(honeycomb_resized, axis=1)
#birdnest_x = tf.keras.utils.normalize(birdnest_resized, axis=1)
#lighthouse_x = tf.keras.utils.normalize(lighthouse_resized, axis=1)

honey_X = extract_rgb_features(honeycomb_resized)
bird_X = extract_rgb_features(birdnest_resized)
light_X = extract_rgb_features(lighthouse_resized)

honey_X_g = extract_glcm_features_for_set(honeycomb_resized_gray, True)
bird_X_g = extract_glcm_features_for_set(birdnest_resized_gray, True)
light_X_g = extract_glcm_features_for_set(lighthouse_resized_gray, True)

honey = np.concatenate((honey_X, honey_X_g), axis=1)
bird = np.concatenate((bird_X, bird_X_g), axis=1)
light = np.concatenate((light_X, light_X_g), axis=1)

honeycomb_y = make_y_for_x(honey, 0)
birdnest_y = make_y_for_x(bird, 1)
lighthouse_y = make_y_for_x(light, 2)

data_x = np.concatenate((honey, bird, light), axis=0)
data_x = stats.zscore(data_x, axis=0, ddof=0)
data_y = np.concatenate((honeycomb_y, birdnest_y, lighthouse_y), axis=0)
data_y_enc = tf.keras.utils.to_categorical(data_y)

data = np.concatenate((data_x, data_y_enc), axis=1)

data = np.concatenate((data_x, data_y_enc), axis=1)
np.random.shuffle(data)

col = data.shape[1]

callback = tf.keras.callbacks.EarlyStopping(monitor='acc', min_delta=0, patience=3, verbose=0, mode='auto')

model = build_model(data_x.shape[1])
model.fit(data[:,:col-3], data[:,[col-3, col-2, col-1]], epochs=500, callbacks = [callback])


