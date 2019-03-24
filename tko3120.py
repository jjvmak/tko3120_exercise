# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:07:54 2019

@author: jjvmak
"""

import numpy as np
from skimage import io
from skimage.transform import resize, rescale, rotate, setup, warp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from skimage.color import rgb2gray
from skimage import img_as_ubyte as eight_bit
from skimage.feature import greycomatrix, greycoprops
import statistics
from sklearn.linear_model import RidgeClassifier


def load_ref_image():
    try:
        image = io.imread(
            'https://images.freeimages.com/images/large-previews/e71/frog-1371919.jpg')
        image_rez, gray_rez = resize_images([image], 300, 300)

    except:
        print("An exception occurred")
    return gray_rez[0]


def create_image_array(location):
    """Returns array of images of given locations."""
    images = []
    urls = np.loadtxt(location, dtype='U100')
    print('reading urls from ' + location)
    print(repr(len(urls)) + ' urls found')
    print()
    # for dev decrease the range! TODO: change back to len(array)
    for i in range(2):
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


def make_X_and_y_df(data, label):
    df = pd.DataFrame(data)
    df['label'] = label
    return df


def make_color_group(data):
    group = []
    for i in range(len(data)):
        if data[i] == 'honey':
            group.append(1)
        if data[i] == 'light':
            group.append(2)
        if data[i] == 'bird':
            group.append(3)
    return group


def c_index(true_labels, predicted_labels):
    n = 0
    h_sum = 0
    for i in range(len(true_labels)):
        t = true_labels[i]
        p = predicted_labels[i]
        j = i + 1
        for j in range(j, len(true_labels)):
            nt = true_labels[j]
            np = predicted_labels[j]
            if t != nt:
                n = n + 1
                if (p < np and t < nt) or (p > np and t > nt):
                    h_sum = h_sum + 1
                elif p == np:
                    h_sum = h_sum + 0.5
    return h_sum / n


def pca_knn_and_cv(data):
    data_x = data.loc[:, data.columns != 'label']
    data_x = data_x.reset_index(drop=True)
    data_x = stats.zscore(data_x, axis=1, ddof=0)
    data_y = data['label']
    data_y = data_y.reset_index(drop=True)

    pca = PCA(n_components=2)
    pca.fit(data_x)
    pca_x = pca.transform(data_x)
    group = make_color_group(data_y)

    plt.scatter(pca_x[:, 0], pca_x[:, 1], c=group)
    plt.show

    for k in range(2, 10):
        predictions = []
        true_labels = []

        for i in range(len(pca_x)):
            train_x = np.delete(pca_x, i, axis=0)
            train_y = data_y.drop(data_y.index[i])
            train_y = train_y.reset_index(drop=True)
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(train_x, train_y)
            result = neigh.predict([pca_x[i]])
            predictions.append(result[0])
            true_labels.append(data_y[i])

        print('c_index: ' + repr(c_index(true_labels, predictions)) + ' k:' + repr(k))
        n = 0
        for j in range(len(predictions)):
            if predictions[j] == true_labels[j]:
                n = n + 1
        print('prediction rate: ' + repr(n) + '/' + repr(len(predictions)))
        print()


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


def ridge_regression_classification(data):
    data_x = data.loc[:, data.columns != 'label']
    data_x = data_x.reset_index(drop=True)
    data_x = stats.zscore(data_x, axis=1, ddof=0)
    data_y = data['label']
    data_y = data_y.reset_index(drop=True)

    pca = PCA(n_components=25)
    pca.fit(data_x)
    pca_x = pca.transform(data_x)

    alpha_for = 0.0

    for a in range(0, 15):
        predictions = []
        true_labels = []

        for i in range(len(pca_x)):
            train_x = np.delete(pca_x, i, axis=0)
            train_y = data_y.drop(data_y.index[i])
            train_y = train_y.reset_index(drop=True)
            clf = RidgeClassifier(alpha=alpha_for, solver='sag', fit_intercept=True)
            clf.fit(train_x, train_y)
            result = clf.predict([pca_x[i]])
            predictions.append(result[0])
            true_labels.append(data_y[i])

        print('c_index: ' + repr(c_index(true_labels, predictions)) + ' alpha:' + repr(alpha_for))
        n = 0
        for j in range(len(predictions)):
            if predictions[j] == true_labels[j]:
                n = n + 1
        print('prediction rate: ' + repr(n) + '/' + repr(len(predictions)))
        print()
        alpha_for = alpha_for + 0.5


# load reference image for glmc
ref_img = load_ref_image()

# data import and resizing pictures
# use 300x300 picture size
x = 300
y = 300
honeycomb_resized, honeycomb_resized_gray = resize_images(create_image_array('honeycomb.txt'), x, y)
birdnest_resized, birdnest_resized_gray = resize_images(create_image_array('birdnests.txt'), x, y)
lighthouse_resized, lighthouse_resized_gray = resize_images(create_image_array('lighthouse.txt'), x, y)

# data preparing and feature extractions

honey_X = extract_rgb_features(honeycomb_resized)
bird_X = extract_rgb_features(birdnest_resized)
light_X = extract_rgb_features(lighthouse_resized)

h_df = make_X_and_y_df(honey_X, 'honey')
b_df = make_X_and_y_df(bird_X, 'bird')
l_df = make_X_and_y_df(light_X, 'light')

data = pd.concat([h_df, b_df, l_df])
data = data.reset_index(drop=True)

honey_X_g = extract_glcm_features_for_set(honeycomb_resized_gray, True)
bird_X_g = extract_glcm_features_for_set(birdnest_resized_gray, True)
light_X_g = extract_glcm_features_for_set(lighthouse_resized_gray, True)

h_df_g = pd.DataFrame(honey_X_g)
b_df_g = pd.DataFrame(bird_X_g)
l_df_g = pd.DataFrame(light_X_g)

data_g = pd.concat([h_df_g, b_df_g, l_df_g])
data_g = data_g.reset_index(drop=True)
all_data = pd.concat([data_g, data], axis=1)

# PCA, training KNN and evaluating the model
pca_knn_and_cv(all_data)
#ridge_regression_classification(data)
