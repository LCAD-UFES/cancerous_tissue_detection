# python crop_viewer.py ../squeezeNet/aux_files/cbisddsm_OF10_automatic_cropped_dataset.txt ../../dataset/cancer_tissue_dataset/automatic_cropped_dataset/

import os, sys
import cv2
import glob
import argparse
from pathlib import Path

# FUNCAO PARA CAPTURAR A POSICAO DO CLICK DO MOUSE ##
# IMPRIME A POSICAO DO CLICK NA TELA ##

parser = argparse.ArgumentParser()
parser.add_argument('--directory', dest = 'directory', help = 'Directory containing images to be processed.', 
    required = True, type = str)
# parser.add_argument('--fileblack', dest = 'fileblack', help = 'The file containing the name of all processed black images.',
#     required = True, nargs='?', type=argparse.FileType('w'), default=sys.stdout)
# parser.add_argument('--filewhite', dest = 'filewhite', help = 'The file containing the name of all processed black images.',
#     required = True, nargs='?', type=argparse.FileType('w'), default=sys.stdout)

args = parser.parse_args()

directory = args.directory
# fileB = args.fileblack
# fileW = args.filewhite

def click_events(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        colorsB = cropImage[y,x,0]
        colorsG = cropImage[y,x,1]
        colorsR = cropImage[y,x,2]
        colors = cropImage[y,x]
        # print(cropImage.dtype)
        print("Red = ", colorsR)
        print("Green = ", colorsG)
        print("Blue = ", colorsB)
        print("BGRformat = ", colors)
        print("Coordinates of pixel: Y = ", y, "X = ", x)

    elif event == cv2.EVENT_MOUSEMOVE:
        color_2 = cropImage[y,x]
        print("Coordinates of pixel: Y = ", y, "X = ", x)
        print("Pixel value = ", color_2)
        # print(cropImage.dtype)



image_paths = Path(os.path.join(directory)).rglob('*.png')
   
    
for image_path in image_paths:

    image_path = str(image_path)

    cropImage = cv2.imread(image_path) 
    # cropImage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(cropImage.shape)

    cv2.namedWindow("crop_image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("crop_image", click_events)

    while (1):
        # cv2.imshow("crop_image", cv2.resize(cropImage, dsize=(224,224), interpolation=cv2.INTER_NEAREST))
        cv2.imshow("crop_image", cropImage)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows() # close displayed windows

