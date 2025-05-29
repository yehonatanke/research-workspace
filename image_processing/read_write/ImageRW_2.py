# open university - image processing course
import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('cromatisity.jpg',cv2.IMREAD_COLOR)
print 'Just read a picture'

plt.imshow(img)

plt.title('cromaticity')
plt.show()

cv2.imwrite('cromaticity.png',img)
