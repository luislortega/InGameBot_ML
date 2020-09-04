#this code show the result in gray scales
import cv2 as cv
import numpy as np

#import of images
haystack_img = cv.imread('albion_farm.jpg', cv.IMREAD_REDUCED_COLOR_2)
needle_img = cv.imread('cabbage.jpg', cv.IMREAD_REDUCED_COLOR_2)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
#multidimentional array with the data 

#[[-0.03693614 -0.01733347  0.00187275 ... -0.10432676 -0.09961629
#  -0.07827686]
# [-0.07594264 -0.06410409 -0.03129069 ... -0.10211685 -0.09682433
#  -0.08718099]
# [-0.08998041 -0.09126714 -0.05543974 ... -0.10111544 -0.10546029
#  -0.10265154]
# ...
# [ 0.17939964  0.17995472  0.20211    ...  0.18741025  0.20025873
#   0.20985427]
# [ 0.15480845  0.16250251  0.1960496  ...  0.16597497  0.17956161
#   0.18149436]
# [ 0.14085224  0.1510219   0.17097141 ...  0.16942734  0.17414725
#   0.17431433]]
print(result)