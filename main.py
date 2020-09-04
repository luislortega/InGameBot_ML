#this code show the result in gray scales
import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#import of images
haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_REDUCED_COLOR_2)
needle_img = cv.imread('cabbage.jpg', cv.IMREAD_REDUCED_COLOR_2)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
##print(result)

threshold = 0.80

#the np.where() return value will look like this:
# (array([175], dtype=int32), array([300], dtype=int32))
locations = np.where(result >= threshold)

print(locations)

#zipping those up into position tuples
locations = list(zip(*locations[::-1]))
print(locations)