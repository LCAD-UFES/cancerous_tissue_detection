import pandas as pd
import numpy as np
import cv2 as cv2
import os, sys, csv, math


def crop_cancer(pathList, roiPathList, labelList):
    j = 0
    while j < len(pathList):
        
        tempImg = roiPathList[j] 
        image = cv2.imread(tempImg)
        tempImg2 = pathList[j]
        examImage = cv2.imread(tempImg2) 
        filename = pathList[j].split('_dataset/')
        croppedImagesPath =  '/mnt/externo/new_dataset/'
        unchekPath = 'UNCHECK/'
        aux = filename[1].split('.')
        patientFile = aux[0]
        sobrepos = 128
        
        #informations about segmentation image 
        smallestBlackPixelAtCoordinateX = 0               
        biggestBlackPixelAtCoordinateX = image.shape[1]            
        biggestBlackPixelAtCoordinateY = image.shape[0]    
        matrixWithThePositionOfTheWhitePixels = np.where(image == [255])
        smallestWhitePixelOfCoordinateX = matrixWithThePositionOfTheWhitePixels[1].min()        
        biggestWhitePixelOfCoordinateX = matrixWithThePositionOfTheWhitePixels[1].max()         
        smallestWhitePixelOfCoordinateY = matrixWithThePositionOfTheWhitePixels[0].min()       
        biggestWhitePixelOfCoordinateY = matrixWithThePositionOfTheWhitePixels[0].max()     
        distanceBetweenSmallestAndBiggestWhitePixelInCoordinateX = biggestWhitePixelOfCoordinateX - smallestWhitePixelOfCoordinateX
        distanceBetweenSmallestAndBiggestWhitePixelInCoordinateY = biggestWhitePixelOfCoordinateY - smallestWhitePixelOfCoordinateY 

        numberOfCroppedsX = int(math.ceil(distanceBetweenSmallestAndBiggestWhitePixelInCoordinateX/(256 - sobrepos)))
        numberOfCroppedsY = int(math.ceil(distanceBetweenSmallestAndBiggestWhitePixelInCoordinateY/(256 - sobrepos)))

        if labelList[j] == True:
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

            savesExamImageWithBoundingBoxes = False

            if distanceBetweenSmallestAndBiggestWhitePixelInCoordinateX <= 256 and distanceBetweenSmallestAndBiggestWhitePixelInCoordinateY <= 256:
                midpointBetweenTheBiggestAndSmallestWhitePixelX = int(distanceBetweenSmallestAndBiggestWhitePixelInCoordinateX/2) + smallestWhitePixelOfCoordinateX   
                midpointBetweenTheBiggestAndSmallestWhitePixelY = int(distanceBetweenSmallestAndBiggestWhitePixelInCoordinateY/2) + smallestWhitePixelOfCoordinateY   
                temporarySmallestWhitePixelOfCoordinateX = midpointBetweenTheBiggestAndSmallestWhitePixelX - 128  
                temporarySmallestWhitePixelOfCoordinateY = midpointBetweenTheBiggestAndSmallestWhitePixelY - 128
                temporaryBiggestWhitePixelOfCoordinateX = midpointBetweenTheBiggestAndSmallestWhitePixelX + 128  
                temporaryBiggestWhitePixelOfCoordinateY = midpointBetweenTheBiggestAndSmallestWhitePixelY + 128
                
                if temporaryBiggestWhitePixelOfCoordinateX > biggestBlackPixelAtCoordinateX:
                    cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX]) 

                
                elif temporarySmallestWhitePixelOfCoordinateX < smallestBlackPixelAtCoordinateX:
                    cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), smallestWhitePixelOfCoordinateX:(smallestWhitePixelOfCoordinateX + 256)])   
            
            
                elif temporarySmallestWhitePixelOfCoordinateX > smallestBlackPixelAtCoordinateX and temporaryBiggestWhitePixelOfCoordinateX < biggestBlackPixelAtCoordinateX:
                    cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), temporarySmallestWhitePixelOfCoordinateX:temporaryBiggestWhitePixelOfCoordinateX])   
                
                cv2.rectangle(mammo_resized, ((int) (temporarySmallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (temporaryBiggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)
                cv2.rectangle(roi_resized, ((int) (temporarySmallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (temporaryBiggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)
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
                        while labelList[j] == False:
                            if labelList[j] == False:
                                j-=1
                        break 
            else:
                temporarySmallestWhitePixelOfCoordinateY = smallestWhitePixelOfCoordinateY
                temporaryBiggestWhitePixelOfCoordinateY = smallestWhitePixelOfCoordinateY + 256
                
                for line in range(int(numberOfCroppedsY)): 
                    temporarySmallestWhitePixelOfCoordinateX = smallestWhitePixelOfCoordinateX
                    temporaryBiggestWhitePixelOfCoordinateX = smallestWhitePixelOfCoordinateX + 256
                    
                    for col in range(int(numberOfCroppedsX)):       
                        
                        if (col == numberOfCroppedsX - 1) and temporaryBiggestWhitePixelOfCoordinateX >= biggestBlackPixelAtCoordinateX:
                            
                            if (biggestWhitePixelOfCoordinateX - temporarySmallestWhitePixelOfCoordinateX) <= 192 :
                                numberOfWhitePixelsInTheRegionOfCropRoiImage = (int)(np.count_nonzero(image[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX]))
                                numberOfWhitePixelsInTheRegionOfCropExamImage = (int)(np.count_nonzero(examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX]))
                                
                                if numberOfWhitePixelsInTheRegionOfCropRoiImage >= 19661 and numberOfWhitePixelsInTheRegionOfCropExamImage >= 19661: #se tiver mais que 70% de pixels brancos, salva normalmente
                                    if (temporaryBiggestWhitePixelOfCoordinateX - sobrepos) > biggestBlackPixelAtCoordinateX: 
                                        cv2.imwrite(os.path.join(croppedImagesPath + unchekPath +str(patientFile) + str(line) + str(col) + '(UNCHECK)' + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX])
                                        cv2.rectangle(mammo_resized, ((int) ((biggestWhitePixelOfCoordinateX - 256) * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) ((biggestWhitePixelOfCoordinateX) * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)
                                        cv2.rectangle(roi_resized, ((int) ((biggestWhitePixelOfCoordinateX - 256) * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) ((biggestWhitePixelOfCoordinateX) * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)
                                        savesExamImageWithBoundingBoxes = True
                                elif numberOfWhitePixelsInTheRegionOfCropRoiImage < 19661 and numberOfWhitePixelsInTheRegionOfCropRoiImage > 9831: #se tiver mais que 15% e menos de 30% de pixels brancos, salva como não checada
                                    if numberOfWhitePixelsInTheRegionOfCropExamImage < 19661 and numberOfWhitePixelsInTheRegionOfCropExamImage > 9831:
                                        cv2.imwrite(os.path.join(croppedImagesPath + unchekPath + str(patientFile) + str(line) + str(col) + '(UNCHECK)' + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX])
                                        cv2.rectangle(mammo_resized, ((int) ((biggestWhitePixelOfCoordinateX - 256) * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) ((biggestWhitePixelOfCoordinateX) * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)
                                        cv2.rectangle(roi_resized, ((int) ((biggestWhitePixelOfCoordinateX - 256) * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) ((biggestWhitePixelOfCoordinateX) * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)
                                        savesExamImageWithBoundingBoxes = True
                            
                            else: 
                                numberOfWhitePixelsInTheRegionOfCropRoiImage = np.count_nonzero(image[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX])
                                numberOfWhitePixelsInTheRegionOfCropExamImage = np.count_nonzero(examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX])
                                
                                if numberOfWhitePixelsInTheRegionOfCropRoiImage >= 19661 and numberOfWhitePixelsInTheRegionOfCropExamImage >= 19661:    #se tiver mais que 70% de pixels brancos, salva normalmente
                                    if (temporaryBiggestWhitePixelOfCoordinateX < biggestBlackPixelAtCoordinateX): #and (biggestWhitePixelOfCoordinateX - (temporarySmallestWhitePixelOfCoordinateX + sobrepos)) > 64:
                                            cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + str(line) + str(col) + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX])
                                            cv2.rectangle(mammo_resized, ((int) ((biggestWhitePixelOfCoordinateX - 256) * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) ((biggestWhitePixelOfCoordinateX) * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)
                                            cv2.rectangle(roi_resized, ((int) ((biggestWhitePixelOfCoordinateX - 256) * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) ((biggestWhitePixelOfCoordinateX) * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)
                                        
                                elif numberOfWhitePixelsInTheRegionOfCropRoiImage < 19661 and numberOfWhitePixelsInTheRegionOfCropRoiImage > 9831: #se tiver mais que 15% e menos de 30% de pixels brancos, salva como não checada
                                    
                                    if numberOfWhitePixelsInTheRegionOfCropExamImage < 19661 and numberOfWhitePixelsInTheRegionOfCropExamImage > 9831:
                                        cv2.imwrite(os.path.join(croppedImagesPath + unchekPath + str(patientFile) + str(line) + str(col) + '(UNCHECK)' + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:(temporarySmallestWhitePixelOfCoordinateY + 256), (biggestWhitePixelOfCoordinateX - 256):biggestWhitePixelOfCoordinateX])
                                        cv2.rectangle(mammo_resized, ((int) ((biggestWhitePixelOfCoordinateX - 256) * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) ((biggestWhitePixelOfCoordinateX) * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)
                                        cv2.rectangle(roi_resized, ((int) ((biggestWhitePixelOfCoordinateX - 256) * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) ((biggestWhitePixelOfCoordinateX) * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)
                                        savesExamImageWithBoundingBoxes = True
                        
                        elif (temporaryBiggestWhitePixelOfCoordinateX - temporarySmallestWhitePixelOfCoordinateX) == 256:
                            numberOfWhitePixelsInTheRegionOfCropRoiImage = np.count_nonzero(image[temporarySmallestWhitePixelOfCoordinateY:temporaryBiggestWhitePixelOfCoordinateY, temporarySmallestWhitePixelOfCoordinateX:temporaryBiggestWhitePixelOfCoordinateX])
                            numberOfWhitePixelsInTheRegionOfCropExamImage = np.count_nonzero(examImage[temporarySmallestWhitePixelOfCoordinateY:temporaryBiggestWhitePixelOfCoordinateY, temporarySmallestWhitePixelOfCoordinateX:temporaryBiggestWhitePixelOfCoordinateX])
                            
                            if numberOfWhitePixelsInTheRegionOfCropRoiImage >= 19661 and numberOfWhitePixelsInTheRegionOfCropExamImage >= 19661:    
                                if (temporaryBiggestWhitePixelOfCoordinateX < biggestBlackPixelAtCoordinateX): #and (biggestWhitePixelOfCoordinateX - (temporarySmallestWhitePixelOfCoordinateX + sobrepos)) > 64:
                                        cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + str(line) + str(col) + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:temporaryBiggestWhitePixelOfCoordinateY, temporarySmallestWhitePixelOfCoordinateX:temporaryBiggestWhitePixelOfCoordinateX]) 
                                        cv2.rectangle(mammo_resized, ((int) (temporarySmallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (temporaryBiggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)        
                                        cv2.rectangle(roi_resized, ((int) (temporarySmallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (temporaryBiggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 255, 0), thickness=1)        
                                        

                            elif numberOfWhitePixelsInTheRegionOfCropRoiImage < 19661 and numberOfWhitePixelsInTheRegionOfCropRoiImage > 9831: #se tiver mais que 15% e menos de 30% de pixels brancos, salva como não checada
                                if numberOfWhitePixelsInTheRegionOfCropExamImage < 19661 and numberOfWhitePixelsInTheRegionOfCropExamImage > 9831:    
                                    cv2.imwrite(os.path.join(croppedImagesPath + unchekPath + str(patientFile) + str(line) + str(col) + '(UNCHECK)' + '.png'), examImage[temporarySmallestWhitePixelOfCoordinateY:temporaryBiggestWhitePixelOfCoordinateY, temporarySmallestWhitePixelOfCoordinateX:temporaryBiggestWhitePixelOfCoordinateX]) 
                                    cv2.rectangle(mammo_resized, ((int) (temporarySmallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (temporaryBiggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)        
                                    cv2.rectangle(roi_resized, ((int) (temporarySmallestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporarySmallestWhitePixelOfCoordinateY * scale_percent / 100)), ((int) (temporaryBiggestWhitePixelOfCoordinateX * scale_percent / 100), (int) (temporaryBiggestWhitePixelOfCoordinateY * scale_percent / 100)), (0, 0, 255), thickness=1)        
                                    savesExamImageWithBoundingBoxes = True
                            
                        cv2.namedWindow(mammo)
                        cv2.moveWindow(mammo, 200, 0)
                        cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        cv2.waitKey(100)                  
                        temporarySmallestWhitePixelOfCoordinateX = temporarySmallestWhitePixelOfCoordinateX + 256 - sobrepos 
                        temporaryBiggestWhitePixelOfCoordinateX = temporaryBiggestWhitePixelOfCoordinateX + 256 - sobrepos   
                    temporarySmallestWhitePixelOfCoordinateY = temporarySmallestWhitePixelOfCoordinateY + 256 - sobrepos
                    temporaryBiggestWhitePixelOfCoordinateY = temporaryBiggestWhitePixelOfCoordinateY + 256 - sobrepos
                
                if savesExamImageWithBoundingBoxes == True:
                    cv2.imwrite(os.path.join(croppedImagesPath + unchekPath + str(patientFile) + '.png'), mammo_resized)   
        
                while True:
                    key = cv2.waitKey(1)
                    if key == 110: # N to next
                        j+=1
                        break
                    elif key == 98: # B to back
                        j-=1
                        while labelList[j] == False:
                            if labelList[j] == False:
                                j-=1
                        break 
                    
        elif labelList[j] == False:
            j+=1      

        cv2.destroyAllWindows() # close displayed windows  
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
            labelList.append(True)
        elif "BENIGN" in i:
            labelList.append(False)
    
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



