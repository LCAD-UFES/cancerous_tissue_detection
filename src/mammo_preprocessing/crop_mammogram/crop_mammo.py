# python crop_mammo.py /mnt/dadosSabrina/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/crop_mammo_val_32 aux_files/test_dataset.txt 32

import cv2
import sys, os
import numpy as np


def image_crop_full_mammo(pathFolder, imageName, imagePath, step):
           
    Y_Max, X_Max, channels = imagePath.shape
    print(imageName)
    
    aux_fileName = imageName.split('_set/') 
    print(aux_fileName)
 
    fileName = aux_fileName[1].split('.')

    label = fileName[0].split('CC_')
    auxLabel = label[1]

    print("FILE  = " + fileName[0])

    print("Y = " + str(Y_Max))
    print("X = " + str(X_Max))

    if label[1] == "MALIGNANT":
        auxLabel = '1'
        print("LABEL = 1" )

    if label[1] == "BENIGN":
        auxLabel = '0'
        print("LABEL = 0" )

    if label[1] == "BENIGN_WITHOUT_CALLBACK":
        auxLabel = '0'
        print("LABEL = 0" )

    # exit()

        
    scale_percent = 15 # percent of original size
    width = int(imagePath.shape[1] * scale_percent / 100)
    height = int(imagePath.shape[0] * scale_percent / 100)
    dim = (width, height)

    
    resized = cv2.resize(imagePath, dim, interpolation = cv2.INTER_AREA)
    

    # classificada = imagePath
    # classificada_resized = cv2.resize(classificada, dim, interpolation = cv2.INTER_AREA)

    # print(fileName[0])

    total_cropy = 0



    for i in range(0, Y_Max, step):
        for j in range(0, X_Max, step):
            cropped_img = imagePath[i:i+256, j:j+256]
            
            if ((cropped_img == 0).all() or cropped_img.shape != (256,256,3)):
                break
            else:
                total_cropy += 1
                cv2.imwrite((pathFolder + '/' + fileName[0] +'_'+ 'Crop_'+ str(total_cropy) + '.png'), cropped_img)
                
                with open(str(pathFolder) + '.txt', 'a+') as myfile:
                    myfile.write(pathFolder + '/' + fileName[0] +'_'+ 'Crop_'+ str(total_cropy) + '.png' + ' ' + auxLabel + '\n')
                    # myfile.write(renamedImagesPath + str(fileName) + '_' + 'BIRADS-' +  str(biRADS) + '_' + 'sub-' + str(difficulty) + '.png\n')
                

                # resized_copy = resized.copy()
                # cv2.rectangle(resized_copy, ((int)(j*scale_percent/100), (int)(i*scale_percent/100)), 
                #                             ((int)((j+256)*scale_percent/100), (int)((i+256)*scale_percent/100)),
                #                             (0,255,0), thickness=3)
                # # cv2.rectangle(resized_copy, ((int)(j*scale_percent/100), (int)(i*scale_percent/100)), 
                # #                             ((int)((j+256)*scale_percent/100), (int)((i+256)*scale_percent/100)),
                # #                             (0,255,0), thickness=3)
                
                

                

                # window_cropped = 'CROP_'+fileName[0]
                # window_original = 'ORIGINAL_'+fileName[0]
                
                # cv2.namedWindow(window_cropped)
                # cv2.moveWindow(window_cropped, 0, 150)
                # cv2.imshow(window_cropped, cropped_img)
                
                # cv2.namedWindow(window_original)
                # cv2.moveWindow(window_original, 700, 150)
                # cv2.imshow(window_original, resized)
                # cv2.waitKey(100)

    print('total_cropy = ', total_cropy)
    
    # print("[Correct marker area] = ", (cancer_area) )
    
    cv2.destroyAllWindows() # close displayed windows



def main(args):
    pathFolder = str(args[1])
    fileName = str(args[2])
    step = int(args[3])

    if os.path.isdir(pathFolder) == False:
        os.mkdir(pathFolder)
    
    file = open(fileName)
    
    for line in file:
        
        str_aux = line.split('\n')
        
        aux = str_aux[0].split(',')
        
        imageName = aux[0]

        print(pathFolder)
        print(fileName)
        print(step)
        print(line)
        print(str_aux)
        print(aux)
        print(imageName)
        print(type(imageName))

        imagePath = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

        print(type(imagePath))

        cv2.imshow('IMAGEM', imagePath)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()


        
        image_crop_full_mammo(pathFolder, imageName, imagePath, step)
    
    file.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv)) 


