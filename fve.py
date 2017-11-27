"""
@donblob
Copyright(C) 2017 donblob, donblob@posteo.org

This program is free software: you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# load images
img = cv2.imread('/path/to/input/image', 0)
# blur image to reduce noise
img = cv2.medianBlur(img, 5)

# equalize the histogram to improve contrast
img = cv2.equalizeHist(img)
# create Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)

# set global threshold value to eliminate grey values (binary)
ret, th1 = cv2.threshold(cl1, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# median to reduce noise
median = cv2.medianBlur(th3, 3)
# blur to smooth edges
blur = cv2.GaussianBlur(median, (3, 3), 0)

# some experimental settings
kernel = np.ones((3, 3), np.uint8)
erosion = cv2.dilate(median, kernel, iterations=1)
opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
opening2 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

# save to disk
cv2.imwrite('/path/to/output/image', th2)

# List with all images and titles
titles = ['Original Image with CLAHE', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Gauss Noise Reduction',
            'blurred and reduced', 'erode', 'morph open']
images = [cl1, th1, th2, th3, median, blur, erosion, closing]
# print stuff
for i in range(8):
    plt.subplot(4, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


