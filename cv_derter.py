import cv2
import sys
import numpy as np
from glob import glob


def vi(fn):
	img = cv2.imread(fn)
	cv2.imshow("input", img)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray", gray)

	blur = cv2.GaussianBlur(gray,(3,3),0)
	cv2.imshow("blur", blur)

	canny = cv2.Canny(blur,0,100,3)
	cv2.imshow("canny", canny)

	contours, hierarchy  = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(img,contours,-1,(255,0,0),2)
	cv2.imshow("contur", img)

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 100, apertureSize=3)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 100, cv2.THRESH_BINARY)
            
            contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):          
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

for fn in glob('./IMG*.JPG'):
    img = cv2.imread(fn)
    vi(fn)
    squares = find_squares(img)
    cv2.drawContours( img, squares, -1, (0, 255, 0), 2 )
    cv2.imshow('squares', img)
    ch = 0xFF & cv2.waitKey()
    if ch == 27:
        break

cv2.destroyAllWindows()
 
