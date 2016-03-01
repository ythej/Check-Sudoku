import argparse
import cPickle
import glob
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required = True,
	help = "Path to the image that contains the sudoku puzzle")

args = vars(ap.parse_args())

image_sudoku_original = cv2.imread(args["path"])

image_sudoku_gray = cv2.cvtColor(image_sudoku_original,cv2.COLOR_BGR2GRAY)
image_sudoku_gray = cv2.GaussianBlur(image_sudoku_gray,(5,5),0)
thresh = cv2.adaptiveThreshold(image_sudoku_gray,255,1,1,11,2)


#find the countours 
_ ,contours0, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#size of the image (height, width)
h, w = image_sudoku_original.shape[:2]

#copy the original image to show the posible candidate
image_sudoku_candidates = image_sudoku_original.copy()

biggest = None
max_area = 0
for i in contours0:
        area = cv2.contourArea(i)
        if area > 100:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area



#show the best candidate
big_rectangle = biggest
approximation = biggest
for i in range(len(approximation)):
    cv2.line(image_sudoku_candidates,
             (big_rectangle[(i%4)][0][0], big_rectangle[(i%4)][0][1]), 
             (big_rectangle[((i+1)%4)][0][0], big_rectangle[((i+1)%4)][0][1]),
             (255, 0, 0), 2)

#show image
# cv2.imshow("Yo", image_sudoku_candidates) 
# cv2.waitKey(0)

import numpy as np
IMAGE_WIDHT = 100
IMAGE_HEIGHT = 100
SUDOKU_SIZE= 9
N_MIN_ACTVE_PIXELS = 10

#sort the corners to remap the image
def getOuterPoints(rcCorners):
    ar = [];
    ar.append(rcCorners[0,0,:]);
    ar.append(rcCorners[1,0,:]);
    ar.append(rcCorners[2,0,:]);
    ar.append(rcCorners[3,0,:]);
    
    x_sum = sum(rcCorners[x, 0, 0] for x in range(len(rcCorners)) ) / len(rcCorners)
    y_sum = sum(rcCorners[x, 0, 1] for x in range(len(rcCorners)) ) / len(rcCorners)
    
    def algo(v):
        return (math.atan2(v[0] - x_sum, v[1] - y_sum)
                + 2 * math.pi) % 2*math.pi
        ar.sort(key=algo)
    return ( ar[0], ar[1], ar[2], ar[3])


#point to remap
points1 = np.array([
                    np.array([0.0,0.0] ,np.float32) + np.array([900,0], np.float32),
                    np.array([0.0,0.0] ,np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([0.0,900], np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([900,900], np.float32),
                    ],np.float32)    
outerPoints = getOuterPoints(approximation)
points2 = np.array(outerPoints,np.float32)

#Transformation matrix
pers = cv2.getPerspectiveTransform(points2,  points1 );

#remap the image
warp = cv2.warpPerspective(image_sudoku_original, pers, (SUDOKU_SIZE*IMAGE_HEIGHT, SUDOKU_SIZE*IMAGE_WIDHT));
warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

warp_sudoku_gray = cv2.GaussianBlur(warp_gray,(5,5),0)
cv2.imshow("Yo", warp_sudoku_gray) 
cv2.waitKey(0)

# def extract_number(x, y):
#     #square -> position x-y
#     im_number = warp_gray[x*IMAGE_HEIGHT:(x+1)*IMAGE_HEIGHT+15, y*IMAGE_WIDHT:(y+1)*IMAGE_WIDHT+15]

#     #threshold
#     #im_number = cv2.GaussianBlur(im_number,(5,5),0)
#     im_number_thresh = cv2.adaptiveThreshold(im_number,255,1,1,15,9)
#     #delete active pixel in a radius (from center) 
#     for i in range(im_number.shape[0]):
#         for j in range(im_number.shape[1]):
#             dist_center =  (IMAGE_WIDHT/2 - i)**2  + (IMAGE_HEIGHT/2 - j)**2;
#             if dist_center > 4900:
#                 im_number_thresh[i,j] = 0;

#     n_active_pixels = cv2.countNonZero(im_number_thresh)
#     return [im_number, im_number_thresh, n_active_pixels]


import Image
import pytesseract
import pytesser

#print "jjjj"+pytesser.mat_to_string(warp_sudoku_thresh)

window_length = 100

# for i in range(9):
# 	for j in range(9):
#  		window = warp_sudoku_thresh[i*window_length: (i+1)*window_length + 10, j*window_length: (j+1)*window_length+10]
#  		cv2.imshow("yaya", window)
#  		cv2.waitKey(0);
#  		print str(i)+" " +str(j)+","+ pytesser.mat_to_string(window)


# for i in range(900):
# 	for j in range(900):
# 		if(warp_sudoku_thresh[i, j] >= 128):
# 			#print "jey"
# 			warp_sudoku_thresh[i, j] = 0
# 		else:
# 			warp_sudoku_thresh[i, j] = 255
# 			#print "ju"


 		# window = warp_sudoku_thresh[i*window_length: (i+1)*window_length + 10, j*window_length: (j+1)*window_length+10]
 		# cv2.imshow("yaya", window)
 		# cv2.waitKey(0);
 		# print str(i)+" " +str(j)+","+ pytesser.mat_to_string(window)


detector = cv2.SimpleBlobDetector_create();


for i in range(9):
	for j in range(9):
 		window = warp_sudoku_gray[i*window_length: (i+1)*window_length +15 , j*window_length: (j+1)*window_length+15]
 		window = cv2.adaptiveThreshold(window, 255,1,1,11,2)

        keypoints = detector.detect(window);
        window = cv2.drawKeypoints(window, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imwrite(str(i)+"_"+str(j)+"."+"jpg", window)
        temp = Image.open(str(i)+"_"+str(j)+"."+"jpg")
        print str(i)+" " +str(j)+","+ pytesseract.image_to_string(temp)



