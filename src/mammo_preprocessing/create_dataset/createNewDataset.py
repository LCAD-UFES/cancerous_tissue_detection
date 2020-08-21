import cv2, argparse
import os, sys, csv
from PIL import Image
import subprocess

def generate_dataset_with_new_names(root, pathList, roiPathList, subList, labelList, numList, typeList):
    j = 0
    if j == 0:
        path = pathList[j].split('_')
        pathName = path[0]
        if pathName == 'Calc-Test':
            ImagesPath = root + 'CBIS-DDSM/CALC/Calc-Test-Full/CBIS-DDSM'
            roiImagesPath = root + 'CBIS-DDSM/CALC/Calc-Test-ROI/CBIS-DDSM'
            if os.path.isdir(root + 'CBIS-DDSM/new-dataset/' + 'Calc-Test') == False:
                os.mkdir(root + 'CBIS-DDSM/new-dataset/' + 'Calc-Test')
                f = open(root + 'CBIS-DDSM/new-dataset/Calc-Test/' + 'Calc-Test.txt', 'w')
                f.close()

        elif pathName == 'Calc-Training':
            ImagesPath = root + 'CBIS-DDSM/CALC/Calc-Training-Full/CBIS-DDSM'
            roiImagesPath = root + 'CBIS-DDSM/CALC/Calc-Training-ROI/CBIS-DDSM'
            if os.path.isdir(root + 'CBIS-DDSM/new-dataset/' + 'Calc-Training') == False:
                os.mkdir(root + 'CBIS-DDSM/new-dataset/' + 'Calc-Training')
                f = open(root + 'CBIS-DDSM/new-dataset/Calc-Training/' + 'Calc-Training.txt', 'w')
                f.close()
                
        elif pathName == 'Mass-Test':
            ImagesPath = root + 'CBIS-DDSM/MASS/Mass-Test-Full/CBIS-DDSM'
            roiImagesPath = root + 'CBIS-DDSM/MASS/Mass-Test-ROI/CBIS-DDSM'
            if os.path.isdir(root + 'CBIS-DDSM/new-dataset/' + 'new-dataset/Mass-Test') == False:
                os.mkdir(root + 'CBIS-DDSM/new-dataset/' + 'Mass-Test')
                f = open(root + 'CBIS-DDSM/new-dataset/Mass-Test/' + 'Mass-Test.txt', 'w')
                f.close()
                        
        elif pathName == 'Mass-Training':
            ImagesPath = root + 'CBIS-DDSM/MASS/Mass-Training-Full/CBIS-DDSM'
            roiImagesPath = root + 'CBIS-DDSM/MASS/Mass-Training-ROI/CBIS-DDSM'
            if os.path.isdir(root + 'CBIS-DDSM/new-dataset/' + 'Mass-Training') == False:
                os.mkdir(root + 'CBIS-DDSM/new-dataset/' + 'Mass-Training')

                f = open(root + 'CBIS-DDSM/new-dataset/Mass-Training/' + 'Mass-Training.txt', 'w')
                f.close()  

    while j < len(pathList):
        if typeList[j] == 'CC':
            tempImg = ImagesPath + '/' + pathList[j]
            examImageBGR = cv2.imread(tempImg)
            examImage =cv2.cvtColor(examImageBGR, cv2.COLOR_BGR2GRAY)
            
            roiPath = roiPathList[j].split('\n')
            roiPath = roiPath[0] 
            tempR = roiImagesPath + '/' + roiPath
            examRoi = cv2.imread(tempR)
            aux = pathList[j].split('/')
            fileName = aux[0]
            biRADS = labelList[j]
            difficulty = subList[j]
            num = numList[j]
            renamedImagesPath = root + 'CBIS-DDSM/new-dataset/' + pathName + '/'

            biggestPixelAtCoordinateX = examImage.shape[1]            
            biggestPixelAtCoordinateY = examImage.shape[0]
            cv2.imwrite(os.path.join(renamedImagesPath + str(fileName) + '_' + 'BIRADS-' + str(biRADS) + '_' + 'sub-' + str(difficulty) + '(' + str(num) + ')' + '.png'), examImage[0:biggestPixelAtCoordinateY,0:biggestPixelAtCoordinateX]) 
        
            biggestPixelAtCoordinateX = examRoi.shape[1]            
            biggestPixelAtCoordinateY = examRoi.shape[0]
            cv2.imwrite(os.path.join(renamedImagesPath + str(fileName) + '_' + 'BIRADS-' + str(biRADS) + '_' + 'sub-' + str(difficulty) + '_' + 'ROI'+ '(' + str(num) + ')' + '.png'), examRoi[0:biggestPixelAtCoordinateY,0:biggestPixelAtCoordinateX]) 

            with open(renamedImagesPath + str(pathName) + '.txt', 'a+') as myfile:
                myfile.write(renamedImagesPath + str(fileName) + '_' + 'BIRADS-' +  str(biRADS) + '_' + 'sub-' + str(difficulty) + '(' + str(num) + ')' + '.png\n')
        j+=1
    return j


#Passar os parâmetros: o caminho até o dataset CBIS-DDSM dentro da máquina e o nome do arquivo .txt a ser lido  


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="the path of CBIS-DDSM")
    parser.add_argument("f", type=str, help="the path of txt file")
    args = parser.parse_args()
    p = "/home/wintermute/" 
    f = "/home/wintermute/breast_cancer_analyzer_LCAD/src/mammo_marker/input_files/calc_case_description_test_set.txt"


    print(args.p)
    print(args.f)
    root = args.p
    if os.path.isdir(root + 'CBIS-DDSM/' + 'new-dataset') == False:
        os.mkdir(root + 'CBIS-DDSM/' + 'new-dataset')
    with open(args.f) as f:
        data = f.readlines()
    reader = csv.reader(data, delimiter=',')
    pathList = []
    roiPathList = []
    labelList = []
    subList = []
    numList = []
    typeList = []
    for row in reader:
        pathList.append((row[11]))
        roiPathList.append((row[12]))
        labelList.append((row[8]))
        subList.append((row[10]))
        numList.append((row[4]))
        typeList.append((row[3]))


    generate_dataset_with_new_names(root, pathList, roiPathList, subList, labelList, numList, typeList)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv)) 