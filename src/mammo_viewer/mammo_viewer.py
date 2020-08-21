# python mammo_viewer.py aux_files/test_viewer_dataset.txt


import cv2
import sys
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn 



def image_tester(imageName, image_GT, imagePath, net):
           
    Y_Max, X_Max, channels = imagePath.shape

    # print(imageName)
    
    aux_fileName = imageName.split('_dataset/') 
    # print(aux_fileName)
 
    fileName = aux_fileName[1].split('.')
    
    # print(fileName)

    # exit()
        
    scale_percent = 10 # percent of original size
    width = int(imagePath.shape[1] * scale_percent / 100)
    height = int(imagePath.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(imagePath, dim, interpolation = cv2.INTER_AREA)
    
    resized_GT = cv2.resize(image_GT, dim, interpolation = cv2.INTER_AREA)
    aux_resized_GT = resized_GT.copy()
    
#     classificada = np.zeros(shape = [Y_Max, X_Max, channels], dtype = np.uint8)
    classificada = imagePath
    classificada_resized = cv2.resize(classificada, dim, interpolation = cv2.INTER_AREA)

    print(fileName[0])

    total_cropy, cancer_cropy, nocancer_cropy,cancer_area = 0, 0, 0, 0

    for i in range(0, Y_Max, 256):
        for j in range(0, X_Max, 256):
            cropped_img = imagePath[i:i+256, j:j+256]
            cropped_tensor = cropped_img.copy()

            cropped_GT = image_GT[i:i+256, j:j+256]           
            
            if ((cropped_img == 0).all() or cropped_img.shape != (256,256,3)):
                break
            else:
                total_cropy += 1
                resized_copy = resized.copy()
                cv2.rectangle(resized_copy, ((int)(j*scale_percent/100), 
                                         (int)(i*scale_percent/100)), ((int)((j+256)*scale_percent/100),
                                                                       (int)((i+256)*scale_percent/100)), (255,0,0), 2)
                
                # cv2.imwrite(('cropped/'+fileName[0] +'_'+ str(i)+'_'+ str(j) + '.png'), cropped_img)
                
                input_class = network_classifier(cropped_tensor, net, fileName[0])
                
                
                # with open('cropp_prob/'+ fileName[0], "a") as input_net:
                #     input_net.write(fileName[0] +'_'+ str(i)+'_'+ str(j) +'\t'+ str(input_class[0][0]) +'\t' + str(input_class[0][1]) + '\n')
                
#                 print(input_class[0][0], input_class[0][1])
                
                
                if input_class[0][1] > 0.93: #COM_cancer
                    cancer_cropy += 1
                    cv2.rectangle(classificada_resized, ((int)(j*scale_percent/100), 
                                         (int)(i*scale_percent/100)), ((int)((j+256)*scale_percent/100),
                                                                       (int)((i+256)*scale_percent/100)), (0,0,255), thickness=3)
                    
                    cv2.rectangle(aux_resized_GT, ((int)(j*scale_percent/100), 
                                         (int)(i*scale_percent/100)), ((int)((j+256)*scale_percent/100),
                                                                       (int)((i+256)*scale_percent/100)), (0,0,255), thickness=3)
                    
                    cv2.imwrite(('../../dataset/classified/'+ fileName[0] + '.png'), classificada_resized)

                    # if (np.any(cropped_GT, ) == 255):
                    #     print("area de segmentacao")
                        
                    #     cancer_area += 1
                        # print(cancer_area)


                    
                # else: #SEM_cancer
                #     nocancer_cropy += 1
                    
                #     cv2.rectangle(classificada_resized, ((int)(j*scale_percent/100), 
                #                          (int)(i*scale_percent/100)), ((int)((j+256)*scale_percent/100),
                #                                                        (int)((i+256)*scale_percent/100)), (0,250,0), thickness=1)
                    
                #     cv2.rectangle(aux_resized_GT, ((int)(j*scale_percent/100), 
                #                          (int)(i*scale_percent/100)), ((int)((j+256)*scale_percent/100),
                #                                                        (int)((i+256)*scale_percent/100)), (0,250,0), thickness=1)

                #     cv2.imwrite(('../../dataset/classified/'+ fileName[0] + '.png'), classificada_resized)
                    

                window_cropped = 'CROP_'+fileName[0]
                window_gt = 'GT_CLASSIFIED_'+fileName[0]
                window_classified = 'CLASSIFIED_'+fileName[0]
                
                cv2.namedWindow(window_cropped)
                cv2.moveWindow(window_cropped, 0, 150)
                cv2.imshow(window_cropped, cropped_img)

                # cv2.namedWindow("cropped_GT")
                # cv2.moveWindow("cropped_GT", 0, 500)
                # cv2.imshow("cropped_GT", cropped_GT)
                
                cv2.namedWindow(window_gt)
                cv2.moveWindow(window_gt, 330, 150)
                cv2.imshow(window_gt, aux_resized_GT)
                
                cv2.namedWindow(window_classified)
                cv2.moveWindow(window_classified, 700, 150)
                cv2.imshow(window_classified, classificada_resized)
                cv2.waitKey(100)

    print('total_cropy, cancer_cropy, nocancer_cropy', total_cropy, cancer_cropy, nocancer_cropy)
    print("Accuracy [cancer_cropy] = ", ((cancer_cropy*100)/total_cropy) )
    print("Accuracy [NOcancer_cropy] = ", ((nocancer_cropy*100)/total_cropy) )
    # print("[Correct marker area] = ", (cancer_area) )
    
    cv2.destroyAllWindows() # close displayed windows
        

def network_classifier(cropped_tensor, net, fileName):
    net.eval()
    cropped_tensor = np.transpose(cropped_tensor, [2, 0, 1])[[2, 1, 0]]
    cropped_tensor = cropped_tensor/255
    cropped_tensor = torch.from_numpy(cropped_tensor.astype(np.float32))
    normalize = transforms.Normalize([0.4818, 0.4818, 0.4818], [0.1752, 0.1752, 0.1752]) #normalização da base
    #normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) #testes antigos
    cropped_tensor = normalize(cropped_tensor)
    cropped_tensor_list = np.array([cropped_tensor.tolist()])
    cropped_tensor = torch.from_numpy(cropped_tensor_list.astype(np.float32))
    
#     print(cropped_tensor)
    
#     torch.set_printoptions(threshold=100000000)
#     
#     with open('cropped_tensor/'+ fileName, "a") as cropped_tensor_file:
#         cropped_tensor_file.write(str(cropped_tensor) + '\n')

    
    with torch.no_grad():
        classification = net(cropped_tensor.to('cuda:0'))
        m = nn.Softmax(dim=1)
        batch_s = m(classification)
        batch_s = batch_s.tolist()
#         print(batch_s)
    
        # for s in batch_s:
        #     with open('probabilities/'+ fileName +'.txt', 'a') as ptest:
        #         ptest.write(str(s[0]) + '\t' + str(s[1]) + '\n')
        
        input_class = batch_s
        
    return input_class
     
def load_matching_name_and_shape_layers(net, new_model_name, new_state_dict):
    # print('\n' + new_model_name + ':')
    state_dict = net.state_dict()
    for key in state_dict:
        if key in new_state_dict and new_state_dict[key].shape == state_dict[key].shape:
            state_dict[key] = new_state_dict[key]
            # print('\t' + key + ' loaded.')
    net.load_state_dict(state_dict)


def Net():
    model = getattr(models, 'squeezenet1_1')
    net = model(num_classes=2)
    return net



def main(args):
    fileName = str(args[1])
    INITIAL_MODEL = str(args[2])
    # INITIAL_MODEL = '/mnt/dadosSabrina/breast_cancer_analyzer_LCAD/src/squeezeNet/squeezenet1_1_89_3.pth' #Sem aumento; Acuracia media = 0.843750000
    # INITIAL_MODEL = '/mnt/dadosSabrina/breast_cancer_analyzer_LCAD/src/squeezetnet/runs/squeezenet1_1/27/models/squeezenet1_1_19_2.pth' #COM aumento; Acuracia da classe 1 = 0.900390625


    torch.multiprocessing.set_start_method('spawn', force=True)
    net = Net().to('cuda:0')
    load_matching_name_and_shape_layers(net, INITIAL_MODEL, torch.load(INITIAL_MODEL))
    
    input_net = open(fileName)
    
    for line in input_net:
        # print(line)
        str_aux = line.split('\n')
        # print(str_aux)
        aux = str_aux[0].split(',')
        # print(aux)
        imageName = aux[0]
        # print(imageName)
        aux_imageGT = aux[1].split('\n')
        # print(aux_imageGT)
        imageGT = aux_imageGT[0]
        # print(imageGT)

        # exit()

        imagePath = cv2.imread(imageName, 1)
        image_GT = cv2.imread(imageGT, 1)
        
        image_tester(imageName, image_GT, imagePath, net)
    
    
    input_net.close()
    

    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv)) 


###python mammo_viewer.py aux_files/test_viewer_dataset.txt