import os
import cv2
import glob
import pydicom
import argparse
import matplotlib.pyplot as plt

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dicom-dir', dest = 'dicom_dir', help = 'Directory containing DICOM images to be processed.', 
    required = True, type = str)
parser.add_argument('-x0', dest = 'x0', help = 'Top-left x-position from where the DICOM image will be erased',
    type = int)
parser.add_argument('-y0', dest = 'y0', help = 'Top-left y-position from where the DICOM image will be erased', 
    type = int)
parser.add_argument('-x1', dest = 'x1', help = 'Bottom-right x-position of the image\'s area to be erased',
    type = int)
parser.add_argument('-y1', dest = 'y1', help = 'Bottom-right y-position of the image\'s area to be erased', 
    type = int)
parser.add_argument('-sh', dest = 'show', help = 'Show image before and after the transformation', action = 'store_true')

args = parser.parse_args()
x0 = args.x0
y0 = args.y0
x1 = args.x1
y1 = args.y1
show = args.show
dicom_dir = args.dicom_dir
if dicom_dir.endswith(os.sep):
    dicom_dir = dicom_dir[:-1]

image_paths = Path(os.path.join(dicom_dir)).rglob('*.dcm')
root_dir = dicom_dir.split(os.sep)[-1]
out_dir = root_dir + '_anonymized'
os.makedirs(out_dir, exist_ok = True)

error = 0
total = 0
for image_path in image_paths:
	image_path = str(image_path)
	image_name = image_path.split(os.sep)[-1]
	out_path = image_path.replace(dicom_dir, out_dir)
	out_path_noimg = out_path.replace(image_name, '')
	os.makedirs(out_path_noimg, exist_ok = True)
	
	print(image_path)
	total += 1
	dcm_image = pydicom.dcmread(image_path)
	try:
		dcm_image.decompress()
	except:
		print('FAILED!')
		error += 1
		continue
	print('\tPatient name before:', dcm_image.PatientName)
	if show:
		plt.title('BEFORE')
		plt.imshow(dcm_image.pixel_array, cmap = plt.cm.bone)
		plt.show()

	dcm_image.PatientName = ''
	pixel_data = dcm_image.pixel_array
	cv2.rectangle(pixel_data, (x0, y0), (x1, y1), 0, cv2.FILLED)
	dcm_image.PixelData = pixel_data.tobytes()
	dcm_image.save_as(out_path)
	print('\tSUCESS: Censure applied!')

	out_image = pydicom.dcmread(out_path)
	print('\tPatient name after:', out_image.PatientName)
	if show:
		plt.title('AFTER')
		plt.imshow(out_image.pixel_array, cmap = plt.cm.bone)
		plt.show()

print('Failed:', error, '/', total)
print(str(100*(1 - error/total)) + '%', 'successful!')
