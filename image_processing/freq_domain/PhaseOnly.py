# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 17:42:56 2015

@author: azariac
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('kidface.jpg',0)



f = np.fft.fft2(img)
f=f/np.abs(f)
img_back = np.fft.ifft2(f)
img_back = np.abs(img_back)
mytlt='Phase only Image'

def myplot(img,img_back,mytlt):
    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title(mytlt), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([]) 
    plt.show()
    return

myplot(img,img_back,mytlt)


