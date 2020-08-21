import cv2, argparse
import os, sys, csv, PIL
from PIL import Image

def generate_dataset_with_new_names(pathList):
    j = 0
    while j < len(pathList):
        tempImg = pathList[j]
        background = cv2.imread('bg.png')
        examImage = cv2.imread(tempImg)
        background[:examImage.shape[0], :examImage.shape[1]]=examImage
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        aux = pathList[j].split('/')
        filename = aux[6]

        # scale_percent = 15 # percent of original size
        # width = int(background.shape[1] * scale_percent / 100)
        # height = int(background.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # mammo_resized = cv2.resize(background, dim, interpolation = cv2.INTER_AREA) 

        # cv2.imshow(filename,mammo_resized)
        cv2.imwrite(os.path.join('/' + aux[0] + '/' + aux[1] + '/' + aux[2] + '/' + aux[3] + '/' + aux[4] + '/' + aux[5] + '/' + str(filename)), background[:,:]) 
        
        # while True:
        #     key = cv2.waitKey(1)
        #     if key == 110: # N to next
        #         j+=1
        #         break
        #     elif key == 98: # B to back
        #         j-=1
        #         break 
        # cv2.destroyAllWindows()
        j+=1
    return j

def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="the path of .txt")
    args = parser.parse_args()
    path = args.p
    with open(args.p) as p:
        data = p.readlines()
    reader = csv.reader(data)
    pathList = []
    for row in reader:
        pathList.append((row[0]))


    generate_dataset_with_new_names(pathList)

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv)) 