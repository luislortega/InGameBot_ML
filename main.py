#this code show the result in gray scales
import cv2 as cv
import numpy as np

#import of images
haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_REDUCED_COLOR_2)
needle_img = cv.imread('cabbage.jpg', cv.IMREAD_REDUCED_COLOR_2)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

#get the best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

#this give the cordinates of the best match in the image
print('Best match top left position: %s' % str(max_loc))
#confidence of the match
print('Best match confidence: %s' % max_val)

#Object found?
threshold = 0.8

if max_val > threshold:
    print('Found')
    
    #get dimension of the needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]
    
    # 
    #  (top_left)
    #       *-------------------*
    #       |                   |
    #       |                   |
    #       |                   |
    #       |                   |
    #       |                   |
    #       *-------------------*
    #                           (bottom_right)
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
    
    #draw the rectangle
    cv.rectangle(haystack_img, top_left, bottom_right, color=(0,255,0), thickness=2, lineType=cv.LINE_4)
    
    #show the result
    cv.imshow('Result', haystack_img)
    cv.waitKey()
else:
    print('Not found')