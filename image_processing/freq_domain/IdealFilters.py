# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 14:09:44 2015

@author: azariac
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('kidface.jpg',0)
img=np.lib.pad(img, (0,1), 'constant', constant_values=(0, 0))


#cv2.imshow("padded image", img)          
#k=cv2.waitKey(0)
#print 'destroy all windows'    
#cv2.destroyAllWindows()



f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fshiftH=np.array(fshift)
fshiftL=np.zeros(fshift.shape)

rows, cols = img.shape
print(img.shape)

crow,ccol = rows/2 , cols/2

print(crow)
print(ccol)
L=10
fshiftH[crow-L:crow+L+1, ccol-L:ccol+L+1] = 0
fshiftL=fshift;
fshiftL=fshift-fshiftH;

f_ishift = np.fft.ifftshift(fshiftH)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
mytlt='Image after HPF'

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

f_ishift = np.fft.ifftshift(fshiftL)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
mytlt='Image after LPF'
myplot(img,img_back,mytlt)

