import pandas as pd
import numpy as np
import cv2 as cv2
import os, sys, csv, math, argparse


def crop_cancer(pathList, roiPathList, labelList, root, folder, sobrepos):
    j = 0
    while j < len(pathList):

        #take roi image 
        tempImg = roiPathList[j] 
        imageBGR = cv2.imread(tempImg, 0)
        image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2GRAY)
        current_all_roi = image

        #take exam image 
        tempImg2 = pathList[j]
        examImageBGR = cv2.imread(tempImg2, 0)
        examImage = cv2.cvtColor(examImageBGR, cv2.COLOR_BGR2GRAY)
        

        filename = pathList[j].split('_dataset/')
        croppedImagesPath =  root + folder + '/'
        aux = filename[1].split('.')
        patientFile = aux[0]
        aux = patientFile.split('_')
        present = aux[0] + aux[1] + aux[2] + aux[3] + aux[4]
        future = " "

        #informations about segmentation image
        if (j+1) < len(pathList):
            nextFile = roiPathList[j+1].split('_dataset/')
            aux = nextFile[1].split('.')
            patientFutureFile = aux[0]
            aux = patientFutureFile.split('_')
            future = aux[0] + aux[1] + aux[2] + aux[3] + aux[4]    
        
        #informations about segmentation image 
        smallestBlackPixelAtCoordinateX = 0
        smallestBlackPixelAtCoordinateY = 0               
        biggestBlackPixelAtCoordinateX = image.shape[1]            
        biggestBlackPixelAtCoordinateY = image.shape[0]  

        if present == future:
            current_all_roi = np.zeros(image.shape)

        
            while (present == future):
                if((j+1) < len(pathList)):
                    nextFile = roiPathList[j+1].split('_dataset/')
                    aux = nextFile[1].split('.')
                    # print(str_aux)
                    patientFutureFile = aux[0]
                    aux = patientFutureFile.split('_')
                    # print(str_aux)
                    future = aux[0] + aux[1] + aux[2] + aux[3] + aux[4]
                else:
                    future = ""
                
                aux = patientFile.split('_')
                present = aux[0] + aux[1] + aux[2] + aux[3] + aux[4]
                # print(str_aux)

                auxRoiImage = roiPathList[j] 
                roiImage = cv2.imread(auxRoiImage, 0)

                current_all_roi+=roiImage
                aux = pathList[j].split('/')
                cv2.imwrite('roi.png', current_all_roi)
                if((j+1)<len(pathList)):
                    j+=1
            
            current_all_roi = cv2.imread('roi.png', 0)
            #print(current_all_roi.shape)

        matrixWithThePositionOfTheWhitePixels = np.where(image == [255])
        smallestWhitePixelOfCoordinateX = matrixWithThePositionOfTheWhitePixels[1].min()        
        biggestWhitePixelOfCoordinateX = matrixWithThePositionOfTheWhitePixels[1].max()         
        smallestWhitePixelOfCoordinateY = matrixWithThePositionOfTheWhitePixels[0].min()       
        biggestWhitePixelOfCoordinateY = matrixWithThePositionOfTheWhitePixels[0].max()     

        numberOfCroppedsX = int(math.floor(biggestBlackPixelAtCoordinateX/224))
        numberOfCroppedsY = int(math.floor(biggestBlackPixelAtCoordinateY/224))

        if labelList[j] == True:
            scale_percent = 15 # percent of original size
            width_roi = int(image.shape[1] * scale_percent / 100)
            height_roi = int(image.shape[0] * scale_percent / 100)
            dim_roi = (width_roi, height_roi)
            roi_resized = cv2.resize(image, dim_roi, interpolation = cv2.INTER_AREA) 
            
            width = int(examImage.shape[1] * scale_percent / 100)
            height = int(examImage.shape[0] * scale_percent / 100)
            dim = (width, height)
            mammo_resized = cv2.resize(examImage, dim, interpolation = cv2.INTER_AREA) 
            mammo = patientFile

            print(mammo)
            
            temporarySmallestBlackPixelOfCoordinateY = smallestBlackPixelAtCoordinateY
            temporaryBiggestBlackPixelOfCoordinateY = smallestBlackPixelAtCoordinateY + 224
            for line in range(int(numberOfCroppedsY)): 
                temporarySmallestBlackPixelOfCoordinateX = smallestBlackPixelAtCoordinateX
                temporaryBiggestBlackPixelOfCoordinateX = smallestBlackPixelAtCoordinateX + 224
                for col in range(int(numberOfCroppedsX)):       
                    if (temporaryBiggestBlackPixelOfCoordinateX > smallestWhitePixelOfCoordinateX and temporaryBiggestBlackPixelOfCoordinateX < biggestWhitePixelOfCoordinateX) and ((temporaryBiggestBlackPixelOfCoordinateY > smallestWhitePixelOfCoordinateY) and (temporaryBiggestBlackPixelOfCoordinateY < biggestWhitePixelOfCoordinateY)):

                        cv2.imwrite(os.path.join(croppedImagesPath + '1' + '_' + str(patientFile) + '_' + str(line) + '_' + str(col) + '_' + '.png'), 
                                    examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                                    temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX])
                        cv2.rectangle(mammo_resized,((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255, 255, 255), thickness=1)        
                        cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255, 255, 255), thickness=1)  
                        with open(croppedImagesPath + 'with_cancer' + '.txt', 'a+') as myfile:
                            myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '1' + '\n')
                        cv2.namedWindow(mammo)
                        cv2.moveWindow(mammo, 200, 0)
                        cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        cv2.waitKey(50)

                    elif (temporarySmallestBlackPixelOfCoordinateX < biggestWhitePixelOfCoordinateX and temporarySmallestBlackPixelOfCoordinateX > smallestWhitePixelOfCoordinateX) and ((temporaryBiggestBlackPixelOfCoordinateY > smallestWhitePixelOfCoordinateY) and (temporaryBiggestBlackPixelOfCoordinateY < biggestWhitePixelOfCoordinateY)):
                        cv2.imwrite(os.path.join(croppedImagesPath + '1' + '_' + str(patientFile) + '_' + str(line) + '_' + str(col) + '_' + '.png'), 
                                    examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                                    temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX])
                        cv2.rectangle(mammo_resized,((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255, 255, 255), thickness=1)        
                        cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255, 255, 255), thickness=1)  
                        with open(croppedImagesPath + 'with_cancer' + '.txt', 'a+') as myfile:
                            myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '1' + '\n')
                        cv2.namedWindow(mammo)
                        cv2.moveWindow(mammo, 200, 0)
                        cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        cv2.waitKey(50)
                    elif (temporaryBiggestBlackPixelOfCoordinateX > smallestWhitePixelOfCoordinateX and temporaryBiggestBlackPixelOfCoordinateX < biggestWhitePixelOfCoordinateX) and ((temporarySmallestBlackPixelOfCoordinateY < biggestWhitePixelOfCoordinateY) and (temporarySmallestBlackPixelOfCoordinateY > smallestWhitePixelOfCoordinateY)):

                        cv2.imwrite(os.path.join(croppedImagesPath + '1' + '_' + str(patientFile) + '_' + str(line) + '_' + str(col) + '_' + '.png'), 
                                    examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                                    temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX])
                        cv2.rectangle(mammo_resized,((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255, 255, 255), thickness=1)        
                        cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255, 255, 255), thickness=1)  
                        with open(croppedImagesPath + 'with_cancer' + '.txt', 'a+') as myfile:
                            myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '1' + '\n')
                        cv2.namedWindow(mammo)
                        cv2.moveWindow(mammo, 200, 0)
                        cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        cv2.waitKey(50)

                    elif (temporarySmallestBlackPixelOfCoordinateX < biggestWhitePixelOfCoordinateX and temporarySmallestBlackPixelOfCoordinateX > smallestWhitePixelOfCoordinateX) and ((temporarySmallestBlackPixelOfCoordinateY < biggestWhitePixelOfCoordinateY) and (temporarySmallestBlackPixelOfCoordinateY > smallestWhitePixelOfCoordinateY)):
                        cv2.imwrite(os.path.join(croppedImagesPath + '1' + '_' + str(patientFile) + '_' + str(line) + '_' + str(col) + '_' + '.png'), 
                                    examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                                    temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX])
                        cv2.rectangle(mammo_resized,((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255, 255, 255), thickness=1)        
                        cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255, 255, 255), thickness=1)  
                        with open(croppedImagesPath + 'with_cancer' + '.txt', 'a+') as myfile:
                            myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '1' + '\n')
                        cv2.namedWindow(mammo)
                        cv2.moveWindow(mammo, 200, 0)
                        cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        cv2.waitKey(50)
                   
                    else:
                        # cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + '_' + str(line) + '_' + str(col) + '.png'), 
                        #                             examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY,
                        #                             temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX]) 
                        # cv2.rectangle(mammo_resized,((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                        #                             (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                        #                             ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                        #                             (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                        #                             (255, 255, 255), thickness=1)        
                        # cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                        #                             (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                        #                             ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                        #                             (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                        #                             (255, 255, 255), thickness=1)        
                        
                        # with open(croppedImagesPath + 'no_cancer' + '.txt', 'a+') as myfile:
                        #     myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '0' + '\n')

                        cv2.namedWindow(mammo)
                        cv2.moveWindow(mammo, 200, 0)
                        cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        cv2.waitKey(50)                  
                    temporarySmallestBlackPixelOfCoordinateX = temporarySmallestBlackPixelOfCoordinateX + 224 - sobrepos 
                    temporaryBiggestBlackPixelOfCoordinateX = temporaryBiggestBlackPixelOfCoordinateX + 224 - sobrepos   
                temporarySmallestBlackPixelOfCoordinateY = temporarySmallestBlackPixelOfCoordinateY + 224 - sobrepos
                temporaryBiggestBlackPixelOfCoordinateY = temporaryBiggestBlackPixelOfCoordinateY + 224 - sobrepos
            j+=1

            # while True:
            #     key = cv2.waitKey(1)
            #     if key == 110: # N to next
            #         j+=1
            #         break
            #     elif key == 98: # B to back
            #         j-=1
            #         while labelList[j] == False:
            #             if labelList[j] == False:
            #                 j-=1
            #         break Test_P_01004_LEFT_CC_2_BENIGN
                 
        
                    
        elif labelList[j] == False:

            if present != future:
                scale_percent = 15 # percent of original size
                width_roi = int(image.shape[1] * scale_percent / 100)
                height_roi = int(image.shape[0] * scale_percent / 100)
                dim_roi = (width_roi, height_roi)
                roi_resized = cv2.resize(image, dim_roi, interpolation = cv2.INTER_AREA) 
                
                width = int(examImage.shape[1] * scale_percent / 100)
                height = int(examImage.shape[0] * scale_percent / 100)
                dim = (width, height)
                mammo_resized = cv2.resize(examImage, dim, interpolation = cv2.INTER_AREA) 
                mammo = patientFile 

                temporarySmallestBlackPixelOfCoordinateY = smallestBlackPixelAtCoordinateY
                temporaryBiggestBlackPixelOfCoordinateY = smallestBlackPixelAtCoordinateY + 224

                            
                for line in range(int(numberOfCroppedsY)): 
                    temporarySmallestBlackPixelOfCoordinateX = smallestBlackPixelAtCoordinateX
                    temporaryBiggestBlackPixelOfCoordinateX = smallestBlackPixelAtCoordinateX + 224
                    
                    for col in range(int(numberOfCroppedsX)):    
                        cv2.imwrite(os.path.join(croppedImagesPath + '0' + '_' + str(patientFile) + '_' + str(line) + '_' + str(col) + '.png'), 
                                        examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX]) 
                        cv2.rectangle(mammo_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                        (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                            ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), (255, 255, 255), thickness=1)        
                        cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), (255, 255, 255), thickness=1)        
                        
                        with open(croppedImagesPath + '/' + 'no_cancer' + '.txt', 'a+') as myfile:
                            myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '0' + '\n')
                
                        cv2.namedWindow(mammo)
                        cv2.moveWindow(mammo, 200, 0)
                        cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        cv2.waitKey(50)                  
                        temporarySmallestBlackPixelOfCoordinateX = temporarySmallestBlackPixelOfCoordinateX + 224 
                        temporaryBiggestBlackPixelOfCoordinateX = temporaryBiggestBlackPixelOfCoordinateX + 224 
                    temporarySmallestBlackPixelOfCoordinateY = temporarySmallestBlackPixelOfCoordinateY + 224 
                    temporaryBiggestBlackPixelOfCoordinateY = temporaryBiggestBlackPixelOfCoordinateY + 224

                j+=1      
            j+=1
        cv2.destroyAllWindows() # close displayed windows  
    return j


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="the path of Dataset")
    parser.add_argument("e", type=str, help="the path of txt file with exams")
    parser.add_argument("r", type=str, help="the path of txt file with roi's")
    parser.add_argument("n", type=str, help="the name of the folder to save the crops")
    parser.add_argument("s", type=int, help="the value of sobreposition")
    args = parser.parse_args()
    
    root = args.p
    folder = args.n
    sobrepos = args.s
    print(args.p)
    print(args.e)
    print(args.r)
    print(args.n)
    print(args.s)
    if os.path.isdir(root + folder) == False:
        os.mkdir(root + folder)

    with open(args.e) as f:
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
    
    with open(args.r) as r:
        roiData = r.readlines()
    roiReader = csv.reader(roiData)
    roiPathList = []
    for row in roiReader:
        roiPathList.append((row[0]))

    crop_cancer(pathList, roiPathList, labelList, root, folder, sobrepos)
    
    return 0
 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 



