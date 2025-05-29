# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 18:42:59 2015

@author: azariac
"""



import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('kidface.jpg',0)
# simple averaging filter without scaling parameter
mean_filter = np.ones((3,3))/9

# creating a guassian filter
x = cv2.getGaussianKernel(5,10)
gaussian = x*x.T

# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
# sobel in x direction
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# laplacian
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', \
                'sobel_y', 'scharr_x']
out_img = [cv2.filter2D(img,-1,kernel) for kernel in filters]

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(out_img[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()