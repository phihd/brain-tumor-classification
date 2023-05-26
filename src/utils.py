import os
import sys

from pathlib import Path
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from sympy import Point, Polygon
from utils import *
from FCM import FCM
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import segmentation_models_pytorch as smp
import pywt

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

sys.path.append(os.path.dirname(os.getcwd()))


def visualize_one_per_class(image_dict, data_path):
    col = 4
    row = int((len(image_dict.keys())/4)) + 1
    fig, ax = plt.subplots(row, col, figsize=(col*5, row*5))
    for i in range(len(image_dict.keys())):
        tumor_type = list(image_dict.keys())[i]
        image_path = os.path.join(data_path, image_dict[tumor_type][0])
        if not os.path.exists(image_path):
            print(image_path)
            break
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        x, y = (int(i%4), int(i/4))
        ax[y, x].imshow(image)
        ax[y, x].title.set_text(tumor_type)
    fig.suptitle('One image per class')
    fig.show()
    fig.savefig('each_class_image.png')
    
    
def visualize_histogram_per_class(image_dict, data_path):
    col = 4
    row = int((len(image_dict.keys())/4)) + 1
    fig, ax = plt.subplots(row, col, figsize=(col*5, row*5))
    for i in range(len(image_dict.keys())):
        tumor_type = list(image_dict.keys())[i]
        count = np.zeros(256)
        for image_name in image_dict[tumor_type]:
            image_path = os.path.join(data_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            hist = np.histogram(image, 255, range=(0, 256))
            count = hist[0]
        bins = hist[1]
        fig = plt.figure()
        x, y = (int(i%4), int(i/4))
        ax[y, x].bar(bins[:-1], count)
        ax[y, x].title.set_text(tumor_type)
        ax[y, x].set_xticks(range(0, 256, 25))
    fig.suptitle('Pixel distribution of each class')
    fig.show()
    fig.savefig('each_class_histogram.png', dpi=150)
 

def contrast_stretching(image, visualize=False):
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    contrasted = cv2.LUT(image, table)
    if visualize:
        plt.figure()
        plt.imshow(contrasted)
        plt.show()
    return contrasted


def visualize_5_largest_contour(image, visualize=False):
    ret, thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    if visualize:
        plt.figure()
        plt.imshow(thresh)
        plt.show()
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE,
                                              method=cv2.CHAIN_APPROX_NONE)

    blank = image.copy()
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    if visualize:
        for i in range(len(contours)):
            contour = contours[i]
            blank = image.copy()
            cv2.polylines(img=blank, pts=[contour],
                                         isClosed=True, color=(255), thickness=2)

            print(cv2.contourArea(contour))
            rect = cv2.boundingRect(contours[i])
            rect_area = rect[2]*rect[3]
            extent = float(cv2.contourArea(contours[i]))/rect_area
            plt.figure()
            plt.imshow(blank)
            plt.show()
            
    threshold = 5
    chosen_contour = None
    square = Polygon(*list(map(Point, [[50, 100], [200, 100], [200, 200], [50, 200]])))
    for i in range(1, len(contours)):
        rect = cv2.boundingRect(contours[i])
        rect_area = rect[2]*rect[3]
        extent = float(cv2.contourArea(contours[i]))/rect_area
        #print(cv2.contourArea(contours[i]))
        #print(extent)
        if (5000 >= cv2.contourArea(contours[i]) >= 1000
                and extent >= 0.5
                and len(Polygon(*list(map(Point, contours[i].reshape(len(contours[i]), 2)))).intersection(square)) > 0):
                #and (cv2.contourArea(contours[i]) * threshold <= cv2.contourArea(contours[i-1]))):
            chosen_contour = contours[i]
            break
    return contours, chosen_contour 


def extract_one_cluster(image, cluster_number):
    return (image == cluster_number).astype(np.uint8)*255





def fuzzy_c_means(image, n_clusters, visualize=False):
    cluster = FCM(image, image_bit=8, n_clusters=n_clusters, m=2, epsilon=0.05, max_iter=100, verbose=False)
    cluster.form_clusters()
    c_means_image = cluster.result

    if visualize:
        fig, ax = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 6))
        for i in range(n_clusters):
            ax[i].imshow(extract_one_cluster(c_means_image, i))
            ax[i].title.set_text('cluster number: ' + str(i))
            
    return c_means_image


def k_means(image, n_clusters, visualize=False):
    vectorized = image.reshape((-1,1))
    kmeans = KMeans(n_clusters=n_clusters, random_state = 0, n_init=5).fit(vectorized)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[kmeans.labels_.flatten()]
    k_means_image = segmented_data.reshape((image.shape))
    
    return k_means_image


def dwt_transform(image, visualize=False):
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    # First level DWT transform
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # Second level DWT transform
    coeffs = pywt.dwt2(LL, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs
    
    if visualize:
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LL, LH, HL, HH]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()
        
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LL2, LH2, HL2, HH2]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()
    
    return LL2