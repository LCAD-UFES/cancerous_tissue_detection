import pandas as pd
import numpy as np
import cv2 as cv2
import os, sys, csv, math


def crop_cancer(pathList, roiPathList, labelList):
    j = 0
    with open('boudingBoxCoord.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL)
        while j < len(pathList):
            
            tempImg = roiPathList[j] 
            image = cv2.imread(tempImg)
            tempImg2 = pathList[j]
            examImage = cv2.imread(tempImg2) 
            filename = pathList[j].split('_dataset/')
            aux = filename[1].split('.')
            patientFile = aux[0]
            
            #informations about segmentation image 
            smallestBlackPixelAtCoordinateX = 0               
            biggestBlackPixelAtCoordinateX = image.shape[1]            
            biggestBlackPixelAtCoordinateY = image.shape[0]    
            matrixWithThePositionOfTheWhitePixels = np.where(image == [255])
            smallestWhitePixelOfCoordinateX = matrixWithThePositionOfTheWhitePixels[1].min()        
            biggestWhitePixelOfCoordinateX = matrixWithThePositionOfTheWhitePixels[1].max()         
            smallestWhitePixelOfCoordinateY = matrixWithThePositionOfTheWhitePixels[0].min()       
            biggestWhitePixelOfCoordinateY = matrixWithThePositionOfTheWhitePixels[0].max()     
            
            scale_percent = 15 # percent of original size
            width_roi = int(image.shape[1] * scale_percent / 100)
            height_roi = int(image.shape[0] * scale_percent / 100)
            dim_roi = (width_roi, height_roi)
            roi_resized = cv2.resize(image, dim_roi, interpolation = cv2.INTER_AREA) 
            roi = patientFile
            
            width = int(examImage.shape[1] * scale_percent / 100)
            height = int(examImage.shape[0] * scale_percent / 100)
            dim = (width, height)
            mammo_resized = cv2.resize(examImage, dim, interpolation = cv2.INTER_AREA) 
            mammo = patientFile

            print(mammo)

            if labelList[j] == 'benign' :
                cv2.rectangle(mammo_resized, ((int) (smallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (smallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (biggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (biggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)
                cv2.rectangle(roi_resized, ((int) (smallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (smallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (biggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (biggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)
            elif labelList[j] == 'malignant' :
                cv2.rectangle(mammo_resized, ((int) (smallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (smallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (biggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (biggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)
                cv2.rectangle(roi_resized, ((int) (smallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (smallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (biggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (biggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)

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
            cv2.destroyAllWindows()
    return j


def main(args):
    examsFile = 'input_files/mamografias_completas.txt'
    roisFile = 'input_files/mamografias_segmentadas.txt'

    with open(examsFile) as f:
        data = f.readlines()
    reader = csv.reader(data)
    pathList = []
    labelList = []
    for row in reader:
        pathList.append((row[0]))

    for i in pathList:
        if "MALIGNANT" in i:
            labelList.append("malignant")
        elif "BENIGN" in i:
            labelList.append("benign")
    
    with open(roisFile) as r:
        roiData = r.readlines()
    roiReader = csv.reader(roiData)
    roiPathList = []
    for row in roiReader:
        roiPathList.append((row[0]))

    crop_cancer(pathList, roiPathList, labelList)
    
    return 0
 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 



