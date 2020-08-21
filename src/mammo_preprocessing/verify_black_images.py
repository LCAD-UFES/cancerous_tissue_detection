#  python verify_black_images.py --directory ../../dataset/cancer_tissue_dataset/automatic_cropped_dataset/ --fileblack blackList.txt --filewhite whiteList.txt

import os, sys
import cv2
import glob
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--directory', dest = 'directory', help = 'Directory containing images to be processed.', 
    required = True, type = str)
parser.add_argument('--fileblack', dest = 'fileblack', help = 'The file containing the name of all processed black images.',
	required = True, nargs='?', type=argparse.FileType('w'), default=sys.stdout)
parser.add_argument('--filewhite', dest = 'filewhite', help = 'The file containing the name of all processed black images.',
	required = True, nargs='?', type=argparse.FileType('w'), default=sys.stdout)

args = parser.parse_args()

directory = args.directory
fileB = args.fileblack
fileW = args.filewhite


if directory.endswith(os.sep):
    directory = directory[:-1]

image_paths = Path(os.path.join(directory)).rglob('*.png')

black = 0
normal = 0
white = 0
total = 0

for image_path in image_paths:
	# print("Processing....")
	image_path = str(image_path)
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	total += 1


	if ((image == 0).all()):
		black += 1
		# print(image_path)
		# print("100% black")
		fileB.write(str(image_path) + '\n')
	elif ((image == 1).all() or (image == 255).all()):
		white += 1
		# print(image_path)
		# print("100% black")
		fileW.write(str(image_path) + '\n')
	else:
		normal += 1
		# print("< 100% black")
		# print(image_path)

fileB.close()
fileW.close()


print("normal = " + str(normal))
print("black = " + str(black))
print("white = " + str(white))
print(normal + black + white)
