import cv2 
import os, sys, argparse

def data_augmentation(path):
    for root, dirs, files in os.walk(path):
        for i, exams in enumerate(files):
            # read image as gray scale
            img = cv2.imread(root + '/' + exams, 0)
            # get image height, width
            (h, w) = img.shape
            # calculate the center of the image
            center = (w / 2, h / 2)
        
            angle90 = 90
            angle180 = 180
            angle270 = 270
            
            scale = 1.0
 
            # Perform the counter clockwise rotation holding at the center
            # 90 degrees
            M = cv2.getRotationMatrix2D(center, angle90, scale)
            rotated90 = cv2.warpAffine(img, M, (h, w))
            
            # 180 degrees
            M = cv2.getRotationMatrix2D(center, angle180, scale)
            rotated180 = cv2.warpAffine(img, M, (w, h))
            
            # 270 degrees
            M = cv2.getRotationMatrix2D(center, angle270, scale)
            rotated270 = cv2.warpAffine(img, M, (h, w))

            #flip vertically
            verticallyFlipped = cv2.flip(img, 0)
            horizontallyFlipped = cv2.flip(img, 1)
            horizontallyAndVerticallyFlipped = cv2.flip(img, -1)

            # create new files
            # cv2.imwrite(root + '/augmented_images/' + '90' + '_' + exams, rotated90)
            # cv2.imwrite(root + '/augmented_images/' + '180' + '_' +  exams, rotated180)
            # cv2.imwrite(root + '/augmented_images/' + '270' + '_' + exams, rotated270)
            # cv2.imwrite(root + '/augmented_images/' + 'verticallyFlipped' + '_' + exams, verticallyFlipped)
            cv2.imwrite(root + '/augmented_images/' + 'horizontallyFlipped' + '_' + exams, horizontallyFlipped)
            cv2.imwrite(root + '/augmented_images/' + 'horizontallyAndVerticallyFlipped' + '_' + exams, horizontallyAndVerticallyFlipped)

    return i

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="the path of Dataset")
    args = parser.parse_args()
    path = args.p
    print(args.p)

    data_augmentation(path)
    
    return 0
 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 





 
