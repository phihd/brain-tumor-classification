from pathlib import Path
import cv2
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from sympy import Point, Polygon
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


