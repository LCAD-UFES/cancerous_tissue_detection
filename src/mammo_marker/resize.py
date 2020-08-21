import pandas as pd
import numpy as np
import cv2 as cv2
import os, sys, csv, math


def resize(pathList, roiPathList):
    j = 0
    while j < len(pathList):
        
        tempImg = roiPathList[j] 
        image = cv2.imread(tempImg)
        tempImg2 = pathList[j]
        examImage = cv2.imread(tempImg2) 
        filename = pathList[j].split('_dataset/')
        roiFilename = roiPathList[j].split('_dataset/')
        resizedImagesPath =  '/mnt/dadosSabrina/breast_cancer_analyzer_LCAD/dataset/original_dataset/resized_dataset/'
        aux = filename[1].split('.')
        patientFile = aux[0]
        aux2 = roiFilename[1].split('.')
        roiPatientFile = aux2[0]
        
        scale_height = 1024
        scale_width = 768

        width_roi = int(image.shape[1]/image.shape[1] * scale_width)
        height_roi = int(image.shape[0]/image.shape[0] * scale_height)
        dim_roi = (width_roi, height_roi)
        roi_resized = cv2.resize(image, dim_roi, interpolation = cv2.INTER_AREA) 
        
        width = int(examImage.shape[1]/examImage.shape[1] * scale_width)
        height = int(examImage.shape[0]/examImage.shape[0] * scale_height)
        dim = (width, height)
        mammo_resized = cv2.resize(examImage, dim, interpolation = cv2.INTER_AREA) 
            
        biggestImagePixelOfCoordinateX = mammo_resized.shape[1]            
        biggestImagePixelOfCoordinateY = mammo_resized.shape[0] 
        biggestRoiPixelOfCoordinateX = roi_resized.shape[1]            
        biggestRoiPixelOfCoordinateY = roi_resized.shape[0] 

        cv2.imwrite(os.path.join(resizedImagesPath + str(roiPatientFile) + '.png'), roi_resized[0:biggestRoiPixelOfCoordinateY, 0:biggestRoiPixelOfCoordinateX]) 
        cv2.imwrite(os.path.join(resizedImagesPath + str(patientFile) + '.png'), mammo_resized[0:biggestImagePixelOfCoordinateY, 0:biggestImagePixelOfCoordinateX]) 

        # cv2.namedWindow(patientFile)
        # cv2.moveWindow(patientFile, 200, 0)
        # cv2.imshow(patientFile, np.hstack([mammo_resized, roi_resized]))
        # cv2.waitKey(100)  
        
        # while True:
        #     key = cv2.waitKey(1)
        #     if key == 110: # N to next
        #         j+=1
        #         break
        #     elif key == 98: # B to back
        #         j-=1

        # cv2.destroyAllWindows() # close displayed windows  
        j+=1
    return j


def main(args):
    examsFile = 'input_files/mamografias_completas.txt'
    roisFile = 'input_files/mamografias_segmentadas.txt'

    with open(examsFile) as f:
        data = f.readlines()
    reader = csv.reader(data)
    pathList = []
    for row in reader:
        pathList.append((row[0]))
    
    with open(roisFile) as r:
        roiData = r.readlines()
    roiReader = csv.reader(roiData)
    roiPathList = []
    for row in roiReader:
        roiPathList.append((row[0]))

    resize(pathList, roiPathList)
    
    return 0
 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 



