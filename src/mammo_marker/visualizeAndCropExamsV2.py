import pandas as pd
import numpy as np
import cv2 as cv2
import argparse
import os, sys, csv, math

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, mammo_resized
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
 
        # draw a rectangle around the region of interest
        cv2.rectangle(mammo_resized, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", mammo_resized)
        print (refPt[0])
        print (refPt[1])


 
# # keep looping until the 'q' key is pressed
# while True:
#     # display the image and wait for a keypress
#     cv2.imshow("image", image)
#     key = cv2.waitKey(1) & 0xFF
 
#     # if the 'r' key is pressed, reset the cropping region
#     if key == ord("r"):
#         image = clone.copy()
 
#     # if the 'c' key is pressed, break from the loop
#     elif key == ord("c"):
#         break
 
# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# if len(refPt) == 2:
#     roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
#     cv2.imshow("ROI", roi)
#     cv2.waitKey(0)

 
# # close all open windows
# cv2.destroyAllWindows()
def crop_cancer(patientList, pathList, subList, labelList):
    j = 0
    while j < len(pathList):
        tempImg = pathList[j]
        examImage = cv2.imread(tempImg) 
        aux = pathList[j].split('/')
        fileName = aux[0]
        patientFile = patientList[j] 
        croppedImagesPath =  '/new_dataset/'
       
        scale_percent = 15 # percent of original size
        width = int(examImage.shape[1] * scale_percent / 100)
        height = int(examImage.shape[0] * scale_percent / 100)
        dim = (width, height)
        mammo_resized = cv2.resize(examImage, dim, interpolation = cv2.INTER_AREA) 
        mammo = patientFile 
       
        
        # load the image, clone it, and setup the mouse callback function
        clone = mammo_resized.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)

        cv2.imwrite(os.path.join(croppedImagesPath + str(fileName) + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX]) 

        cv2.rectangle(mammo_resized, ((int) (temporarySmallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (temporaryBiggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)
        cv2.namedWindow(mammo)
        cv2.moveWindow(mammo, 200, 0)
        cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
        cv2.waitKey(100)  
        
        while True:
            key = cv2.waitKey(1)
            if key == 110: # N to next
                j+=1
                break
            elif key == 98: # B to back
                j-=1
                break 
            

        cv2.destroyAllWindows() # close displayed windows  
    return j



def main(args):
    examsFile = 'input_files/calc_case_description_test_set.txt'

    with open(examsFile) as f:
        data = f.readlines()
    reader = csv.reader(data, delimiter=',')
    patientList = []
    pathList = []
    labelList = []
    subList = []
    for row in reader:
        patientList.append((row[0]))
        pathList.append((row[11]))
        labelList.append((row[8]))
        subList.append((row[10]))


    crop_cancer(patientList, pathList, subList, labelList)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv)) 



