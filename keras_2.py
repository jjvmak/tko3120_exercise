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
import os
import shutil
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte as eight_bit
from skimage.feature import greycomatrix, greycoprops
import statistics


def create_image_array(location):
    images = []
    urls = np.loadtxt(location, dtype='U100')
    #print('reading urls from ' + location)
    #print(repr(len(urls)) + ' urls found')
    #print()
    for i in range(3):
        #print('reading image ' + repr(i) + ' from ' + urls[i])
        try:
            image = io.imread(urls[i])
            images.append(image)
        except:
            print("An exception occurred")
    return images


def resize_images(image_array, x, y):
    resized_images = []
    resized_gray_images = []
    for i in range(len(image_array)):
        img = image_array[i]
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
    # Builds single sequential models with one hidden layer
    # Activation function in hidden layr is relu.
    # Output layers activation is softmax
    # Models is compiled with adam optimizer and categorical crossentropy loss
    # function.

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_dim=data_shape))
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


def split_data(data):
    # Splits data to 3 arrays for models
    splitted = np.array_split(data, 3)
    return splitted[0], splitted[1], splitted[2]


def build_and_save_sub_model(data, dim_for_subs):
    # FOR SOME REASON THIS IS DOES NOT WORK IN JUPYTER NOTEBOOK (no surprises
    # there...)
    # Builds sequential models and saves them.
    # See build_model() for detailed information.
    # Removes pre existing models
    if os.path.exists('models'):
        shutil.rmtree('models')
    os.makedirs('models')

    for i in range(len(data)):
        data_set = data[i]
        model = build_model(dim_for_subs)
        model.fit(data_set[:,:col-3], data_set[:,[col-3, col-2, col-1]], epochs=500,
          callbacks = [callback], validation_split=0.1, verbose=1)
        filename = 'models/model_' + str(i + 1) + '.h5'
        model.save(filename)
        print('>Saved %s' % filename)
        print()


def build_and_save_sub_model_cached_version(data, dim_for_subs):
    models = []
    for i in range(len(data)):
        data_set = data[i]
        model = build_model(dim_for_subs)
        model.fit(data_set[:,:col-3], data_set[:,[col-3, col-2, col-1]], epochs=500,
          callbacks = [callback], validation_split=0.1, verbose=1)
        models.append(model)
        print('added model')
        print()
    return models


def load_all_models(n_models):
    # THIS DOES NOT WORK IN JUPYTER NOTEBOOK

    all_models = list()
    for i in range(n_models):
        # Defines filename for this ensemble.
        filename = 'models/model_' + str(i + 1) + '.h5'
        # Load model from file.
        model = tf.keras.models.load_model(filename)
        # Add to list of members.
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def stack_models(members, test_data):
    # how to do the stacking???
    # There where few tutorials about how to combine models, but none of those
    # did not seem to work with the version of tf.keras which was installed
    # by the course instructions.

    # This functions DOES NOT actually stack models but instead of that, just
    # will get predictions of each model, store the in the arrays and get the
    # mode value of the predictions. This value will be the final prediction
    # of the ensemble.

    # This is kinda redundant now because the stacking did not work.
    # This would prevent changing the weight and bias values of the models.
    subs = []
    for i in range(len(members)):
        sub = members[i]
        for layer in sub.layers:
            # make not trainable
            layer.trainable = False
        subs.append(sub)

    sub_0 = subs[0]
    sub_1 = subs[1]
    sub_2 = subs[2]

    pred_0 = sub_0.predict_classes(test[:,:col-3])
    pred_1 = sub_1.predict_classes(test[:,:col-3])
    pred_2 = sub_2.predict_classes(test[:,:col-3])
    modes = []

    # Getting the predictions.
    # modes is the final predictions of the ensemble.
    for i in range(len(pred_0)):
        mode = stats.mode([pred_0[i], pred_1[i], pred_2[i]], axis=None)
        modes.append(mode[0][0])

    return pred_0, pred_1, pred_2, modes


print()

# reference image for glcm patches
ref_img = load_ref_image()

x = 300
y = 300
honeycomb_resized, honeycomb_resized_gray = resize_images(create_image_array('honeycomb.txt'), x, y)
birdnest_resized, birdnest_resized_gray = resize_images(create_image_array('birdnests.txt'), x, y)
lighthouse_resized, lighthouse_resized_gray = resize_images(create_image_array('lighthouse.txt'), x, y)

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
# Standardize the data
data_x = stats.zscore(data_x, axis=0, ddof=0)

data_y = np.concatenate((honeycomb_y, birdnest_y, lighthouse_y), axis=0)
# Enconding the labels for keras
data_y_enc = tf.keras.utils.to_categorical(data_y)
# Combine and shuffle data. Shuffling will improve weight calculations.
data = np.concatenate((data_x, data_y_enc), axis=1)
np.random.shuffle(data)

# Splitting data to training and testing sets
data, test = train_test_split(data, test_size=0.10, random_state=42)

col = data.shape[1]

# This is the callback function used to early stopping. It will monitor the
# loss metric and stop once it will not improve more than min_delta.
# Patience is set to 2, so it will not stop at the first time we reach
# min_delta but will keep going till it will reach min_delta second time.
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001,
                                            patience=2, verbose=1, mode='auto')

dim_for_subs = data_x.shape[1]

# Splitting data for three different models
s1, s2, s3 = split_data(data)

# Build and save three models
#build_and_save_sub_model([s1, s2, s3], dim_for_subs)

# Load models
#sub_models = load_all_models(3)

# USE CACHED VERSION IN JUPYTER NOTEBOOK
sub_models = build_and_save_sub_model_cached_version([s1, s2, s3], dim_for_subs)

# Get the predictions of the models with the testing data.
pred_0, pred_1, pred_2, pred_3 = stack_models(sub_models, test)

# 'decode' back the encoded labels
true_values = (np.argmax(test[:,[col-3, col-2, col-1]], axis=1))

# Calculate simple performance metrics
# (correct predictions / true values)
correct = 0
for i in range(len(true_values)):
    if true_values[i] == pred_3[i]:
        correct += 1

print()
print('prediction rate: ' + repr(correct / len(true_values)))







