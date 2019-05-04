import numpy as np
import cv2
#from matplotlib import pyplot as plt

img = cv2.imread('../images/basil.jpg')
dst=np.zeros(img.shape[:2],np.float32)
cv2.imshow('b',img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imshow('sure_bg',sure_bg)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#dist_transform1=cv2.normalize(dist_transform,dst)
#print(dist_transform1[200,200])
cv2.imshow('dist_transform',dist_transform)

ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv2.imshow('sure_fg',sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('unknown',unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
cv2.imshow('markers',markers)

# Add one to all labels so that sure background is not 0, but 1
#markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]#分水岭完成后轮廓边界以-1填充
cv2.imshow('img',img)
#cv2.imshow('a',img)
cv2.waitKey()
#plt.imshow(img)
#plt.show()
