#this code show the result in gray scales
import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#import of images
haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_REDUCED_COLOR_2)
needle_img = cv.imread('cabbage.jpg', cv.IMREAD_REDUCED_COLOR_2)

needle_w = needle_img.shape[1]
needle_h = needle_img.shape[0]

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
##print(result)

threshold = 0.50

#the np.where() return value will look like this:
# (array([175], dtype=int32), array([300], dtype=int32))
locations = np.where(result >= threshold)

#print(locations)

#zipping those up into position tuples
locations = list(zip(*locations[::-1]))
#print(locations)

#first we need to create the list of [x,y,w,h] rectangles
rectangles = []

for loc in locations:
    rect = [int(loc[0]), int(loc[1]), needle_w, needle_h]
    rectangles.append(rect)

rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)

print(rectangles)


if len(rectangles):
    print("Found")
    
    #get dimension of the needle image
    for (x,y,w,h) in rectangles:
        #
        # Determine the box positions 
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
        top_left = (x,y)
        bottom_right = (x + w, y +h)

        #draw the rectangle
        cv.rectangle(haystack_img, top_left, bottom_right, color=(0,255,0), thickness=2, lineType=cv.LINE_4)
        
        '''
        Drawing a mark
        '''
        
        center_x = x + int(w/2)
        center_y = y + int(h/2)
        cv.drawMarker(haystack_img, (center_x, center_y), color=(255,0,255), line_type=cv.MARKER_CROSS)

    #show the result
    cv.imshow('Result', haystack_img)
    cv.waitKey()
        
    
else:
    print("Not found")