import cv2 as cv
import sys

img = cv.imread(sys.argv[1], 1)
cv.imshow("original", img)

