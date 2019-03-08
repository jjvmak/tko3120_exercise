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


def create_image_array(location):
    """Returns array of images of given locations."""
    images = []
    urls = np.loadtxt(location, dtype='U100')
    print('reading urls from ' + location)
    print(repr(len(urls)) + ' urls found')
    # for dev decrease the range! TODO: change back to len(array)
    for i in range(len(urls)):
        # print('reading image ' + repr(i) + ' from ' + urls[i])
        try:
           image = io.imread(urls[i])
           images.append(image)
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

# data import and preparation phase
# TODO: what is good ratio for images? 100x100?
# TODO: Reduce the quantization level e.g. to 8 levels (wtf)
x = 50
y = 50
honeycomb_resized = resize_images(create_image_array('honeycomb.txt'), x, y)
birdnest_resized = resize_images(create_image_array('birdnests.txt'), x, y)
lighthouse_resized = resize_images(create_image_array('lighthouse.txt'), x, y)

# TODO: gray scale images

# io.imshow(lighthouse_resized[0])
# io.show()
# io.imshow(birdnest_resized[0])
# io.show()
# io.imshow(honeycomb_resized[0])
# io.show()


# feature extraction
# TODO: maybe do this in function
honey_X = extract_rgb_features(honeycomb_resized)
bird_X = extract_rgb_features(birdnest_resized)
light_X = extract_rgb_features(lighthouse_resized)

h_df = make_X_and_y_df(honey_X, 'honey')
b_df = make_X_and_y_df(bird_X, 'bird')
l_df = make_X_and_y_df(light_X, 'light')

data = pd.concat([h_df, b_df, l_df])
data = data.reset_index(drop=True)
data_X = data.loc[:, data.columns != 'label']
data_X = data_X.reset_index(drop=True)
data_X = stats.zscore(data_X, axis=1, ddof=1)
data_Y = data['label']
data_Y = data_Y.reset_index(drop=True)
pca = PCA(n_components=2)
pca.fit(data_X)
pca_X = pca.transform(data_X)
pca_X_df = pd.DataFrame(pca_X)
group = make_color_group(data_Y)
plt.scatter(pca_X[:,0], pca_X[:,1], c=group)
plt.show


train_X = np.delete(pca_X, 0, axis=0)
train_Y = data_Y.drop(data_Y.index[0])
train_Y = train_Y.reset_index(drop=True)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_X, train_Y)
result = neigh.predict([pca_X[0]])
print(result[0])




