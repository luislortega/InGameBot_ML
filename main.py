#this code show the result in gray scales
import cv2 as cv
import numpy as np

haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('cabbage.jpg', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

#get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

#this give the cordinates of the best match in the image
print('Best match top left position: %s' % str(max_loc))
#confidence of the match
print('Best match confidence: %s' % max_val)