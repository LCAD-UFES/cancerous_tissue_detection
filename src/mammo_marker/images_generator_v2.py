import cv2
import os, sys, csv

def image_generator(examsPath, listOfFiles, j):
    if listOfFiles[j][0] != listOfFiles[j-1][0]:
        tempImg = listOfFiles[j][0]
        image = cv2.imread(tempImg)
        pathLabels = "newLabels/"
        goodPath = "good/new/"
        benignPath = "benign/new/"
        malignantPath = "malignant/new/"
        tempFileName = os.path.basename(listOfFiles[j][0]) 
        fileName = tempFileName.split(".")
        listOfCoordinates = []
        img = os.path.join(pathLabels + fileName[0] + '.txt')
        if os.path.exists(img):
            with open(pathLabels + fileName[0] + '.txt', newline = '') as textFile:
                fileReader = csv.reader(textFile, delimiter = ' ')
                y = 0
                for i in fileReader:
                    listOfCoordinates.append(i)
                    centerX = listOfCoordinates[y][1]
                    centerY = listOfCoordinates[y][2]
                    minX = int(centerX) - 128
                    maxX = int(centerX) + 128
                    minY = int(centerY) - 128
                    maxY = int(centerY) + 128
                    if listOfCoordinates[y][0] == "MALIGNANT":
                            cv2.imwrite(os.path.join(malignantPath + fileName[0] +'_Crop_' + str(y) + '.png'), image[int(minY):int(maxY),int(minX):int(maxX)])
                    elif listOfCoordinates[y][0] == "GOOD":
                            cv2.imwrite(os.path.join(goodPath + fileName[0] +'_Crop_' + str(y) + '.png'), image[int(minY):int(maxY),int(minX):int(maxX)])
                    elif listOfCoordinates[y][0] == "BENIGN":
                            cv2.imwrite(os.path.join(benignPath + fileName[0] +'_Crop_' + str(y) + '.png'), image[int(minY):int(maxY),int(minX):int(maxX)])
                    y+=1
        else:
            print(">>>>>>>>>>>>>>>>>>> ERROR:" + img + " Doesn't exists!")
            exit()

    return j

def main(args):
    examsPath = str(args[1])  
    listOfFiles = []
    with open(examsPath, newline = '') as textFile:
        fileReader = csv.reader(textFile, delimiter = ' ')
        i = 0
        for i in fileReader:
            listOfFiles.append(i)
    crop = [image_generator(examsPath, listOfFiles, j) for j in range(len(listOfFiles))]
    return 0
 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 