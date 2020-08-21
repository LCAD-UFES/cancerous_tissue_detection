import pandas as pd
import numpy as np
import cv2 as cv2
import os, sys, csv, math, argparse


def rotate_exam_and_segmentation(pathList, roiPathList):
    j = 0
    while j < len(pathList):
        
        
        #rotate image to left orientation
        tempImg = roiPathList[j] 
        print(tempImg)
        imageBGR = cv2.imread(tempImg)
        image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2GRAY)
        tempImg2 = pathList[j]
        examImageBGR = cv2.imread(tempImg2)
        examImage = cv2.cvtColor(examImageBGR, cv2.COLOR_BGR2GRAY)
        numberOfWhitePixelsInTheLeftRegion = (np.count_nonzero(examImage[0:examImage.shape[0], 0:int(math.floor(examImage.shape[1]/2))]))
        numberOfWhitePixelsInTheRightRegion = (np.count_nonzero(examImage[0:examImage.shape[0], int(math.floor(examImage.shape[1]/2)):examImage.shape[1]]))
        
        if numberOfWhitePixelsInTheLeftRegion < numberOfWhitePixelsInTheRightRegion:
            #exam
            bgrImageFlip = cv2.imread(tempImg2)
            imageFlip = cv2.cvtColor(bgrImageFlip, cv2.COLOR_BGR2GRAY)
            imageFlip = cv2.flip(imageFlip, 1)
            cv2.imwrite(pathList[j], imageFlip)
            del imageFlip
            #roi
            bgrRoiImageFlip = cv2.imread(tempImg)
            imageFlip = cv2.cvtColor(bgrRoiImageFlip, cv2.COLOR_BGR2GRAY)
            imageFlip = cv2.flip(imageFlip, 1)
            cv2.imwrite(roiPathList[j], imageFlip)
            del imageFlip
        j+=1
        cv2.destroyAllWindows() # close displayed windows  
    return j


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("e", type=str, help="the path of txt file with exams")
    parser.add_argument("r", type=str, help="the path of txt file with roi's")
    args = parser.parse_args()

    with open(args.e) as f:
        data = f.readlines()
    reader = csv.reader(data)
    pathList = []
    for row in reader:
        pathList.append((row[0]))
    
    with open(args.r) as r:
        roiData = r.readlines()
    roiReader = csv.reader(roiData)
    roiPathList = []
    for row in roiReader:
        roiPathList.append((row[0]))

    rotate_exam_and_segmentation(pathList, roiPathList)
    
    return 0
 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 
