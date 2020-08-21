#  python mergeSeg.py /mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/dataset/Calc-CC_flipped_dataset/ /mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/aux_files/mamografias_completas.txt /mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/aux_files/mamografias_segmentadas.txt ../automatic_cropped_dataset 0

import pandas as pd
import numpy as np
import cv2 as cv2
import os, sys, csv, math, argparse


def crop_cancer(pathList, roiPathList, labelList, root, folder, sobrepos):
    j = 0
    while j < len(pathList):
        
        #full mammogram
        auxExamImage = pathList[j]
        # print(auxExamImage)
        examImage = cv2.imread(auxExamImage, 0)

        print(examImage.shape, examImage.max(), examImage.min())

        birads = labelList[j]

        #roi segmented image
        auxRoiImage = roiPathList[j]
        roiImage = cv2.imread(auxRoiImage, 0)
        current_all_roi = roiImage
        print(roiImage.shape, roiImage.max(), roiImage.min())
        

        #informations about full mammogram 
        filename = pathList[j].split('_dataset/')
        # print(filename)
        croppedImagesPath =  root + folder + '/'
        str_aux = filename[1].split('.')
        patientFile = str_aux[0]
        str_aux = patientFile.split('_')
        present = str_aux[0] + str_aux[1] + str_aux[2] + str_aux[3] + str_aux[4]
        future = " "

        #informations about segmentation image
        if (j+1) < len(pathList):
            nextFile = roiPathList[j+1].split('_dataset/')
            str_aux = nextFile[1].split('.')
            patientFutureFile = str_aux[0]
            str_aux = patientFutureFile.split('_')
            future = str_aux[0] + str_aux[1] + str_aux[2] + str_aux[3] + str_aux[4]        
        

        smallestBlackPixelAtCoordinateX = 0
        smallestBlackPixelAtCoordinateY = 0               
        biggestBlackPixelAtCoordinateX = roiImage.shape[1]            
        biggestBlackPixelAtCoordinateY = roiImage.shape[0]

        # print(patientFile + "\t" + patientFutureFile)

        if present == future:
            current_all_roi = np.zeros(roiImage.shape)

        
            while (present == future):
                if((j+1) < len(pathList)):
                    nextFile = roiPathList[j+1].split('_dataset/')
                    str_aux = nextFile[1].split('.')
                    # print(str_aux)
                    patientFutureFile = str_aux[0]
                    str_aux = patientFutureFile.split('_')
                    # print(str_aux)
                    future = str_aux[0] + str_aux[1] + str_aux[2] + str_aux[3] + str_aux[4]
                else:
                    future = ""
                
                str_aux = patientFile.split('_')
                present = str_aux[0] + str_aux[1] + str_aux[2] + str_aux[3] + str_aux[4]
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
        
        
        numberOfCroppedsX = int(math.floor(biggestBlackPixelAtCoordinateX/224))
        numberOfCroppedsY = int(math.floor(biggestBlackPixelAtCoordinateY/224))

        # if labelList[j] == True:
        scale_percent = 10 # percent of original size
        width_roi = int(current_all_roi.shape[1] * scale_percent / 100)
        height_roi = int(current_all_roi.shape[0] * scale_percent / 100)
        dim_roi = (width_roi, height_roi)
        # print(dim_roi)
        roi_resized = cv2.resize(current_all_roi, dim_roi, interpolation = cv2.INTER_AREA) 
        
        width = int(examImage.shape[1] * scale_percent / 100)
        height = int(examImage.shape[0] * scale_percent / 100)
        dim = (width, height)
        # print(dim)
        mammo_resized = cv2.resize(examImage, dim, interpolation = cv2.INTER_AREA) 
        mammo = patientFile

        print("Pacient = " + mammo)
        
        temporarySmallestBlackPixelOfCoordinateY = smallestBlackPixelAtCoordinateY
        temporaryBiggestBlackPixelOfCoordinateY = smallestBlackPixelAtCoordinateY + 224


        for line in range(int(numberOfCroppedsY)): 
            # print(line)
            temporarySmallestBlackPixelOfCoordinateX = smallestBlackPixelAtCoordinateX
            temporaryBiggestBlackPixelOfCoordinateX = smallestBlackPixelAtCoordinateX + 224
            
            for col in range(int(numberOfCroppedsX)):
                # print(col)
                if np.any(current_all_roi[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                    temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX] == [255]):

                    # if ((examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                    #     temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX] == [0]).all()) != True:

                    if (count_zeros(examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                        temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX]) == False):

                        cv2.rectangle(mammo_resized,((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                (0), thickness=2)        
                        cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                (0), thickness=2)
                        # print(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '1')
    
                    
                        
                    
                        cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + '_' + str(line) + '_' + str(col) + '.png'), 
                                examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY,
                                temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX])
                        with open(croppedImagesPath + '../with_cancer' + '.txt', 'a+') as myfile:
                            myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + str(birads) + '\n')
                        
                        # cv2.namedWindow(mammo)
                        # cv2.moveWindow(mammo, 0, 0)
                        # cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        # cv2.waitKey(20)
                
                else:
                    # if ((examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                    #     temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX] == [0]).all()) != True:
                    if (count_zeros(examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
                        temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX]) == False):
                        
                        cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + '_' + str(line) + '_' + str(col) + '.png'), 
                                                    examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY,
                                                    temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX]) 
                        cv2.rectangle(mammo_resized,((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255), thickness=1)        
                        cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
                                                    (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
                                                    (255), thickness=1)        
                        
                        with open(croppedImagesPath + '../no_cancer' + '.txt', 'a+') as myfile:
                            # print(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '0')
                            myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '8' + '\n')

                        # cv2.namedWindow(mammo)
                        # cv2.moveWindow(mammo, 0, 0)
                        # cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
                        # cv2.waitKey(20)                  
                    
                temporarySmallestBlackPixelOfCoordinateX = temporarySmallestBlackPixelOfCoordinateX + 224 - sobrepos
                temporaryBiggestBlackPixelOfCoordinateX = temporaryBiggestBlackPixelOfCoordinateX + 224 - sobrepos
            
            temporarySmallestBlackPixelOfCoordinateY = temporarySmallestBlackPixelOfCoordinateY + 224 - sobrepos
            temporaryBiggestBlackPixelOfCoordinateY = temporaryBiggestBlackPixelOfCoordinateY + 224 - sobrepos
        j+=1
                 
        
                    
        # elif labelList[j] == False:
        #     if present != future:
        #         scale_percent = 15 # percent of original size
        #         width_roi = int(roiImage.shape[1] * scale_percent / 100)
        #         height_roi = int(roiImage.shape[0] * scale_percent / 100)
        #         dim_roi = (width_roi, height_roi)
        #         roi_resized = cv2.resize(roiImage, dim_roi, interpolation = cv2.INTER_AREA) 
                
        #         width = int(examImage.shape[1] * scale_percent / 100)
        #         height = int(examImage.shape[0] * scale_percent / 100)
        #         dim = (width, height)
        #         mammo_resized = cv2.resize(examImage, dim, interpolation = cv2.INTER_AREA) 
        #         mammo = patientFile 

        #         print("Pacient = " + mammo)

        #         temporarySmallestBlackPixelOfCoordinateY = smallestBlackPixelAtCoordinateY
        #         temporaryBiggestBlackPixelOfCoordinateY = smallestBlackPixelAtCoordinateY + 224

                            
        #         for line in range(int(numberOfCroppedsY)): 
        #             temporarySmallestBlackPixelOfCoordinateX = smallestBlackPixelAtCoordinateX
        #             temporaryBiggestBlackPixelOfCoordinateX = smallestBlackPixelAtCoordinateX + 224
                    
        #             for col in range(int(numberOfCroppedsX)):

        #                 if (count_zeros(examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
        #                     temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX]) == False):
                        
        #                 # if ((examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
        #                 #     temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX] == [0]).all()) != True:


        #                     cv2.rectangle(mammo_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
        #                         (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
        #                         ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
        #                         (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
        #                         (255), thickness=1)

        #                     cv2.rectangle(roi_resized, ((int) (temporarySmallestBlackPixelOfCoordinateX * scale_percent / 100), 
        #                         (int) (temporarySmallestBlackPixelOfCoordinateY * scale_percent / 100)), 
        #                         ((int) (temporaryBiggestBlackPixelOfCoordinateX * scale_percent / 100), 
        #                         (int) (temporaryBiggestBlackPixelOfCoordinateY * scale_percent / 100)), 
        #                         (255), thickness=1)

        #                     cv2.imwrite(os.path.join(croppedImagesPath + str(patientFile) + '_' + str(line) + '_' + str(col) + '.png'), 
        #                         examImage[temporarySmallestBlackPixelOfCoordinateY:temporaryBiggestBlackPixelOfCoordinateY, 
        #                         temporarySmallestBlackPixelOfCoordinateX:temporaryBiggestBlackPixelOfCoordinateX])
                            
        #                     with open(croppedImagesPath + '../no_cancer' + '.txt', 'a+') as myfile:
        #                         myfile.write(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '0' + '\n')
        #                         # print(str(patientFile) + '_' + str(line) + '_' + str(col) + '.png' + ' ' + '0' + '\n')
                    
        #                     # cv2.namedWindow(mammo)
        #                     # cv2.moveWindow(mammo, 0, 0)
        #                     # cv2.imshow(mammo, np.hstack([mammo_resized, roi_resized]))
        #                     # cv2.waitKey(20)                  
                            
        #                 temporarySmallestBlackPixelOfCoordinateX = temporarySmallestBlackPixelOfCoordinateX + 224 
        #                 temporaryBiggestBlackPixelOfCoordinateX = temporaryBiggestBlackPixelOfCoordinateX + 224 

        #             temporarySmallestBlackPixelOfCoordinateY = temporarySmallestBlackPixelOfCoordinateY + 224 
        #             temporaryBiggestBlackPixelOfCoordinateY = temporaryBiggestBlackPixelOfCoordinateY + 224

        #         j+=1      
        #     j+=1
        # cv2.destroyAllWindows() # close displayed windows  
    return j

def count_zeros(image):
    h = image.shape[0]
    w = image.shape[1]

    total = h * w

    black = 0
    others = 0

    for y in range(0, h):
        for x in range(0, w):
            if (image[y,x] == 0):
                black +=1
            else:
                others += 1

    # print("black = " + str(black))
    # print("others = " + str(others))

    if black > (total * 0.98):
        # print(black, others)
        return True
    else:
        # print('false')
        # print(black, others)
        return False


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="the path of Dataset")
    parser.add_argument("e", type=str, help="the path of txt file with exams")
    parser.add_argument("r", type=str, help="the path of txt file with roi's")
    parser.add_argument("n", type=str, help="the name of the folder to save the crops")
    parser.add_argument("s", type=int, help="the value of sobreposition")
    args = parser.parse_args()
    # root = '/home/wintermute/breast_cancer_analyzer_LCAD/dataset/new_dataset/' 
    # e = '/home/wintermute/breast_cancer_analyzer_LCAD/dataset/new_dataset/Calc-Test-CC.txt'
    # r = '/home/wintermute/breast_cancer_analyzer_LCAD/dataset/new_dataset/Calc-Test-CC-ROI.txt'
    # folder = 'birads-dataset'
    # sobrepos = 0
    root = args.p
    folder = args.n
    sobrepos = args.s

    print("Path = " + args.p)
    print("Mammograms List = " + args.e)
    print("Segmentation List = " + args.r)
    print("Save Folder = " + args.n)
    print("Sobreposition = " + str(args.s))

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
        if "BIRADS-0" in i:
            labelList.append('0')
        elif "BIRADS-1" in i:
            labelList.append('1')
        elif "BIRADS-2" in i:
            labelList.append('2')
        elif "BIRADS-3" in i:
            labelList.append('3')
        elif "BIRADS-4" in i:
            labelList.append('4')
        elif "BIRADS-5" in i:
            labelList.append('5')

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



