import cv2
import sys
import time


def rotate_image(imagePath):
    # read image as grey scale
    img = cv2.imread(imagePath)
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    
    aux_fileName = imagePath.split('../../dataset/')
    aux2_fileName = aux_fileName[1].split('/') 
    fileName = aux2_fileName[1].split('.')
    
    angle90 = 90
    angle180 = 180
    angle270 = 270
     
    scale = 1.0
    
    #FLIP VERTICAL
    flip_vertical = cv2.flip(img, 0)

    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (h, w))
    flip_rotated90 = cv2.warpAffine(flip_vertical, M, (h, w))
     
    # 180 degrees
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated180 = cv2.warpAffine(img, M, (w, h))
    flip_rotated180 = cv2.warpAffine(flip_vertical, M, (h, w))
     
    # 270 degrees
    M = cv2.getRotationMatrix2D(center, angle270, scale)
    rotated270 = cv2.warpAffine(img, M, (h, w))
    flip_rotated270 = cv2.warpAffine(flip_vertical, M, (h, w))

    
    window_original = 'ORIGINAL_'+fileName[0]
    window_90 = 'ROTATED_90_'+fileName[0]
    window_180 = 'ROTATED_180_'+fileName[0]
    window_270 = 'ROTATED_270_'+fileName[0]
    window_Flip = 'FLIP_VERTICAL_'+fileName[0]
    window_Flip_90 = 'FLIP_ROTATED_90_'+fileName[0]
    window_Flip_180 = 'FLIP_ROTATED_180_'+fileName[0]
    window_Flip_270 = 'FLIP_ROTATED_270_'+fileName[0]


    if aux2_fileName[0] == 'malignant':
        
        print(fileName[0])
        
#         cv2.namedWindow(window_original)
#         cv2.moveWindow(window_original, 0, 0)
#         cv2.imshow(window_original,img)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_90)
#         cv2.moveWindow(window_90, 400, 0)
#         cv2.imshow(window_90,rotated90)
        cv2.imwrite(('../../dataset/augmented_malignant/'+ fileName[0] + '_90D' + '.png'), rotated90)
#         cv2.waitKey(10) # waits until a key is pressed
	    
#         cv2.namedWindow(window_180)
#         cv2.moveWindow(window_180, 800, 0)
#         cv2.imshow(window_180,rotated180)
        cv2.imwrite(('../../dataset/augmented_malignant/'+ fileName[0] + '_180D' + '.png'), rotated180)
#         cv2.waitKey(10) # waits until a key is pressed

#         cv2.namedWindow(window_270)
#         cv2.moveWindow(window_270, 1200, 0)
#         cv2.imshow(window_270,rotated270)
        cv2.imwrite(('../../dataset/augmented_malignant/'+ fileName[0] + '_270D' + '.png'), rotated270)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_Flip)
#         cv2.moveWindow(window_Flip, 0, 400)
#         cv2.imshow(window_Flip,flip_vertical)
        cv2.imwrite(('../../dataset/augmented_malignant/'+ fileName[0] + '_FLIP' + '.png'), flip_vertical)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_Flip_90)
#         cv2.moveWindow(window_Flip_90, 400, 400)
#         cv2.imshow(window_Flip_90,flip_rotated90)
        cv2.imwrite(('../../dataset/augmented_malignant/'+ fileName[0] + '_FLIP90D' + '.png'), flip_rotated90)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_Flip_180)
#         cv2.moveWindow(window_Flip_180, 800, 400)
#         cv2.imshow(window_Flip_180,flip_rotated180)
        cv2.imwrite(('../../dataset/augmented_malignant/'+ fileName[0] + '_FLIP180D' + '.png'), flip_rotated180)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_Flip_270)
#         cv2.moveWindow(window_Flip_270, 1200, 400)
#         cv2.imshow(window_Flip_270,flip_rotated270)
        cv2.imwrite(('../../dataset/augmented_malignant/'+ fileName[0] + '_FLIP270D' + '.png'), flip_rotated270)
#         cv2.waitKey(3000) # waits until a key is pressed
        
#         cv2.destroyAllWindows() # destroys the window showing image

    if aux2_fileName[0] == 'good':
        
        print(fileName[0])
        
#         cv2.namedWindow(window_original)
#         cv2.moveWindow(window_original, 0, 0)
#         cv2.imshow(window_original,img)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_90)
#         cv2.moveWindow(window_90, 400, 0)
#         cv2.imshow(window_90,rotated90)
        cv2.imwrite(('../../dataset/augmented_good/'+ fileName[0] + '_90D' + '.png'), rotated90)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_180)
#         cv2.moveWindow(window_180, 800, 0)
#         cv2.imshow(window_180,rotated180)
        cv2.imwrite(('../../dataset/augmented_good/'+ fileName[0] + '_180D' + '.png'), rotated180)
#         cv2.waitKey(10) # waits until a key is pressed

#         cv2.namedWindow(window_270)
#         cv2.moveWindow(window_270, 1200, 0)
#         cv2.imshow(window_270,rotated270)
        cv2.imwrite(('../../dataset/augmented_good/'+ fileName[0] + '_270D' + '.png'), rotated270)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_Flip)
#         cv2.moveWindow(window_Flip, 0, 400)
#         cv2.imshow(window_Flip,flip_vertical)
        cv2.imwrite(('../../dataset/augmented_good/'+ fileName[0] + '_FLIP' + '.png'), flip_vertical)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_Flip_90)
#         cv2.moveWindow(window_Flip_90, 400, 400)
#         cv2.imshow(window_Flip_90,flip_rotated90)
        cv2.imwrite(('../../dataset/augmented_good/'+ fileName[0] + '_FLIP90D' + '.png'), flip_rotated90)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_Flip_180)
#         cv2.moveWindow(window_Flip_180, 800, 400)
#         cv2.imshow(window_Flip_180,flip_rotated180)
        cv2.imwrite(('../../dataset/augmented_good/'+ fileName[0] + '_FLIP180D' + '.png'), flip_rotated180)
#         cv2.waitKey(10) # waits until a key is pressed
        
#         cv2.namedWindow(window_Flip_270)
#         cv2.moveWindow(window_Flip_270, 1200, 400)
#         cv2.imshow(window_Flip_270,flip_rotated270)
        cv2.imwrite(('../../dataset/augmented_good/'+ fileName[0] + '_FLIP270D' + '.png'), flip_rotated270)
#         cv2.waitKey(3000) # waits until a key is pressed
#         cv2.destroyAllWindows() # destroys the window showing image

def main(args):
    fileName = str(args[1])
    input_net = open(fileName)
    
    start = time.time()
#     print(start)
    for line in input_net:
        aux = line.split('\n')
        imageName = aux[0]
#         print('imageName')
#         print(imageName)
        rotate_image(imageName)

    input_net.close()
    finish = time.time()
#     print(finish)
    print("Tempo de execucao = " + str(finish-start))  
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv)) 

