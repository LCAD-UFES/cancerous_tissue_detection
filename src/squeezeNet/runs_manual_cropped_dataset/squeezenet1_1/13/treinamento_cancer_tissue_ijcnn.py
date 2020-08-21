from __future__ import division, print_function
import os, shutil, time, random
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms

## SALVAR OS TREINOS COM BONS RESULTADOS NO HD EXTERNO##
RUNS_FOLDER = '/home/sabrina/breast_cancer_analyzer_LCAD/src/squeezeNet/runs_manual_cropped_dataset'

NETWORK = 'squeezenet1_1'

NUM_CLASSES = 2

## COLOCAR O CAMINHO DO PESO QUE SERÁ UTILIZADO PARA TESTAR ##
INITIAL_MODEL = None
INITIAL_MODEL_TEST = False 

TRAINING = (
        '/home/sabrina/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/training_dataset_ijcnn.txt', #57344 imagens
)

TRAINING_DIR = (
        '',
)

##USAR TRUE APENAS NO PRIMEIRO TREINO. USAR O MESMO ARQUIVO NOS DEMAIS TREINOS##
SHUFFLE =  False

##USAR APENAS O CONJUNTO DE VALIDACAO DURANTE O TREINO. USAR CONJUNTO DE TESTE NO SCRIPT TEST.PY##
TEST = (
         # '/home/sabrina/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_val_2019_09_12_1.txt', #832 imagens
         '/home/sabrina/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_val_2019_10_15.txt', #7168 imagens
)

TEST_DIR = (
        # '',    
         '/home/sabrina/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/manual_cropped_dataset',
)

TRANSFORMS = transforms.Normalize([0.4818, 0.4818, 0.4818], [0.1752, 0.1752, 0.1752])

BATCH_SIZE, ACCUMULATE = 128, 1

EPOCHS = 40
SAVES_PER_EPOCH = 1

INITIAL_LEARNING_RATE = 0.0003
LAST_EPOCH_FOR_LEARNING_RATE_DECAY = 35
DECAY_RATE = 2
DECAY_STEP_SIZE = 3
##UTILIZAR VALOR 1 QUANDO USAR UTILIZAR APRESENTACAO DAS IMAGENS##
# NUM_WORKERS = 1
NUM_WORKERS = 4

##DESCOMENTAR PARA USAR O CLICK_EVENTS##
# POS_X, POS_Y = 0, 0


def load_matching_name_and_shape_layers(net, new_model_name, new_state_dict):
    print('\n' + new_model_name + ':')
    state_dict = net.state_dict()
    for key in state_dict:
        if key in new_state_dict and new_state_dict[key].shape == state_dict[key].shape:
            state_dict[key] = new_state_dict[key]
            print('\t' + key + ' loaded.')
    net.load_state_dict(state_dict)

def Net():
    model = getattr(models, NETWORK)
    net = model(num_classes=NUM_CLASSES)
    
    #inicializar a rede de forma randomica.
    # load_matching_name_and_shape_layers(net, 'Torchvision pretrained model', model(pretrained=False).state_dict()) 

    #inicializa com imagenet ou peso indicado
    load_matching_name_and_shape_layers(net, 'Torchvision pretrained model', model(pretrained=True).state_dict()) 
    return net

##DEFINICAO DO DATASET##
class DatasetFromCSV(Dataset):
    def __init__(self, csv_files, root_dirs, label=None, shuffle=False, transforms=None, dataset_file=None):
        temp_image = np.zeros((224,224))
        self.norm_image = None

        data = []
        for csv_file, root_dir in zip(csv_files, root_dirs):
            d = pd.read_csv(csv_file, header=None, names=['images', 'labels'], delim_whitespace=True)
            if label != None:
                d = d.loc[d['labels'] == label].reset_index().drop('index', 1)
            if root_dir != '':
                d['images'] = root_dir + ('' if root_dir.endswith('/') else '/') + d['images']
            data.append(d)
        if len(data) == 1:
            data = data[0]
        else:
            data = pd.concat(data).reset_index().drop('index', 1)
        self.data_len = len(data)

        if shuffle:
            perm = np.arange(self.data_len)
            random.shuffle(perm)
            data = data.iloc[perm].reset_index().drop('index', 1)

        self.images = np.asarray(data.iloc[:, 0])
        self.labels = np.asarray(data.iloc[:, 1])
        # print('TRANSFORMS = ', transforms)
        self.transforms = transforms
        # print('TRANSFORMS = ', transforms)

        if dataset_file != None:
            with open(dataset_file, 'w') as dataset:
                for image, label in zip(self.images, self.labels):
                    dataset.write(image + ' ' + str(label) + '\n')

    def __len__(self):
        return self.data_len

    ## COMENTAR O TRECHO ABAIXO CASO QUEIRA APRESENTAR AS IMAGENS EM TELA##
    def __getitem__(self, i):
        image = cv2.imread(self.images[i], 3)
        # print('NOME DA IMAGEM: ', self.images[i])
        # print("\nMATRIZ DA IMAGEM REAL= ", image)
        image = np.transpose(image, [2, 0, 1])[[2, 1, 0]]
        # print("\nMATRIZ TRANSPOSTA DA IMAGEM REAL= ", image)
        image = image/255
        # print("\nMATRIZ DA IMAGEM REAL/255 [RANGE 0-1]  = ", image)
        image = torch.from_numpy(image.astype(np.float32))
        # print("\nTENSOR DA IMAGEM REAL [RANGE 0-1] = ", image)
        if self.transforms != None:
            # print('TRANSFORMS = ', transforms)
            # print("\nTENSOR DA IMAGEM REAL ANTES DE NORMALIZAR = ", image)
            image = self.transforms(image)
            # print('TENSOR DA IMAGEM DEPOIS DE NORMALIZAR: ', image)
        return (image, self.labels[i], self.images[i])

    
#######################################################################################
    ## DESCOMENTAR O TRECHO ABAIXO PARA APRESENTAR AS IMAGENS EM TELA##
    
    # FUNCAO PARA CAPTURAR A POSICAO DO CLICK DO MOUSE ##
    # IMPRIME A POSICAO DO CLICK NA TELA ##
    # def click_events(self, event, x, y, flags, param):
    #     global POS_X
    #     global POS_Y

    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         colors = self.temp_image[y,x]
    #         norm_colors = self.norm_image[y,x]
    #         print()
    #         print("Coordinates of pixel: Y = ", y, "X = ", x)
    #         print("Pixel value = ", colors)
    #         print("Tensor value = ", norm_colors)           
            
            
    #     elif event == cv2.EVENT_MOUSEMOVE:
    #         colors = self.temp_image[y,x]
    #         norm_colors = self.norm_image[y,x]
    #         print()
    #         print("Coordinates of pixel: Y = ", y, "X = ", x)
    #         print("Pixel value = ", colors)
    #         print("Tensor value = ", norm_colors)

    #         # print(image.dtype)


    # def __getitem__(self, i):
    #     global POS_X
    #     global POS_Y
        
    #     image = cv2.imread(self.images[i], 3)
    #     #print('NOME DA IMAGEM: ', self.images[i])
    #     #print('LABEL IMAGEM: ', self.labels[i])

    #     #print("\nMATRIZ DA IMAGEM REAL= ", image)
        
    #     cv2.namedWindow('ENTRADA')
    #     cv2.moveWindow('ENTRADA', 300, 0)
    #     cv2.imshow('ENTRADA', image)
        
    #     self.temp_image = image
        
    #     image = np.transpose(image, [2, 0, 1])[[2, 1, 0]]
    #     #print("\nMATRIZ TRANSPOSTA DA IMAGEM REAL= ", image)
        
    #     image = image/255
    #     #print("\nMATRIZ DA IMAGEM REAL/255 = ", image)
        
    #     image = torch.from_numpy(image.astype(np.float32))
    #     #print("\nTENSOR DA IMAGEM REAL = ", image)

    #     if self.transforms != None:
    #         #print("\n\n NORMALIZATION \n")

    #         image = self.transforms(image)
    #         #print('TENSOR DA IMAGEM DEPOIS DE NORMALIZAR: ', image) 

    #         self.norm_image = np.moveaxis(image.numpy(), 0, 2)

    #         cpy = image.numpy().copy()            
    #         cpy -= np.min(cpy)
    #         cpy /= np.max(cpy)
    #         cpy = np.moveaxis(cpy, 0, 2)

    #         #print(cpy.shape)
    #         #print(cpy.dtype)             

    #         cv2.namedWindow('NORMALIZADA')
    #         cv2.moveWindow('NORMALIZADA', 800, 0)
    #         cv2.setMouseCallback('NORMALIZADA', self.click_events)
        

    #         cv2.imshow('NORMALIZADA', cpy)

            
    #         # cv2.waitKey(500) #automatico
    #         cv2.waitKey(0) #espera tecla

    #     cv2.destroyAllWindows()
    #     #exit()

    #     return (image, self.labels[i], self.images[i])

#######################################################################################


def test(net, dataset_name, datasets_per_label, dataloaders_per_label, results_file=None, classification_error_file=None):
    net.eval()
    str_buf = '\n\t' + dataset_name + ':\n\n\t\tConfusion Matrix\tClass Accuracy\n'
    print(str_buf)
    if results_file != None:
        with open(results_file, 'a') as results:
            results.write(str_buf + '\n')
    average_class_accuracy = 0.0
    valid_classes = 0
    for i in range(NUM_CLASSES):
        dataset, dataloader = datasets_per_label[i], dataloaders_per_label[i]
        line = np.zeros(NUM_CLASSES, dtype=int)
        class_accuracy = 0.0
        if dataset.data_len > 0:
            valid_classes += 1
            with torch.no_grad():
                for batch in dataloader:
                    classification = net(batch[0].to('cuda:0'))
                    # print("classification", classification)
                    # print("Batch: {0}\n, {1}\n, {2}\n ".format(batch[0].shape, batch[1], batch[2]))
                    c = torch.max(classification, 1)[1].tolist()
                    # print("Resultado classificacao",c)
                    # print()
                    
                    # for pred, lbl, filename in zip(c, batch[1], batch[2]):
                    #     # print("Predicao = ", pred)
                    #     # print("Label = ", lbl.item())
                    #     # print("Arquivo = ", filename)
                    #     # print()


                    #     if (lbl.item() == 0) & (pred == 1):
                    #         if classification_error_file != None:
                    #             with open(classification_error_file, 'a') as classification_error:
                    #                 classification_error.write("Falso NEGATIVO[0/1]" + "\n")
                    #                 classification_error.write("Label" + '\t' + "Pred" + '\t' + "PathFile" + "\n")
                    #                 classification_error.write('\t' + str(lbl.item()) + '\t')
                    #                 classification_error.write('\t' + str(pred) + '\t')
                    #                 classification_error.write(filename + '\n')
                    #                 # print("---------------------------")
                    #                 # print("Falso NEGATIVO[0/1]", filename)
                    #                 # print("---------------------------")

                    #     elif (lbl.item() == 1) & (pred == 0):
                    #         if classification_error_file != None:
                    #             with open(classification_error_file, 'a') as classification_error:
                    #                 classification_error.write("Falso POSITIVO[0/1]"+ "\n")
                    #                 classification_error.write("Label" + '\t' + "Pred" + '\t' + "PathFile" + "\n")
                    #                 classification_error.write('\t' + str(lbl.item()) + '\t')
                    #                 classification_error.write('\t' + str(pred) + '\t')
                    #                 classification_error.write(filename + '\n')
                    #                 # print("++++++++++++++++++++++++++")
                    #                 # print("Falso POSITIVO[1/0]", filename)
                    #                 # print("++++++++++++++++++++++++++")
                    
                    for j in range(NUM_CLASSES):
                        line[j] += c.count(j)
                        # print(line[j],c.count(j))
            class_accuracy = float(line[i])/dataset.data_len
            average_class_accuracy += class_accuracy
        str_buf = '\t'
        for j in range(NUM_CLASSES):
            str_buf += '\t' + str(line[j])
        str_buf += '\t\t{:.9f}'.format(class_accuracy)
        print(str_buf)
        if results_file != None:
            with open(results_file, 'a') as results:
                results.write(str_buf + '\n')
    average_class_accuracy /= valid_classes
    str_buf = '\n\t\tAverage Class Accuracy: {:.9f}'.format(average_class_accuracy)
    print(str_buf)
    if results_file != None:
        with open(results_file, 'a') as results:
            results.write(str_buf + '\n')
    net.train()


def main():
    torch.multiprocessing.set_start_method('spawn', force=True)

    net = Net().to('cuda:0')
    if INITIAL_MODEL != None:
        load_matching_name_and_shape_layers(net, INITIAL_MODEL, torch.load(INITIAL_MODEL))


    if TEST != None:
        if INITIAL_MODEL_TEST:
            print('\n' + (INITIAL_MODEL if INITIAL_MODEL != None else 'Initial model') + ' tests:')
        tests = []
        for csv_file, root_dir in zip(TEST, TEST_DIR):
            datasets_per_label = [DatasetFromCSV((csv_file,), (root_dir,), label=i, transforms=TRANSFORMS) for i in range(NUM_CLASSES)]
            dataloaders_per_label = [DataLoader(dataset, BATCH_SIZE, num_workers=NUM_WORKERS) for dataset in datasets_per_label]
            tests.append((csv_file, datasets_per_label, dataloaders_per_label))
            if INITIAL_MODEL_TEST:
                test(net, csv_file, datasets_per_label, dataloaders_per_label)


    if TRAINING == None:
        exit()

    net_folder = os.path.join(RUNS_FOLDER, NETWORK)
    i = 1
    while True:
        save_folder = os.path.join(net_folder, ('0' if i < 10 else '') + str(i))
        if os.path.exists(save_folder):
            i += 1
        else:
            break
    models_folder = os.path.join(save_folder, 'models')
    os.makedirs(models_folder)
    shutil.copy(__file__, save_folder)
    training_dataset_file = os.path.join(save_folder, 'training_dataset.txt')
    training_log_file = os.path.join(save_folder, 'training_log.txt')
    loss_log_file = os.path.join(save_folder, 'loss_log.txt')
    results_file = os.path.join(save_folder, 'results.txt')
    
    classification_error_file = os.path.join(save_folder, 'classification_error.txt')
    
    print('\nSave folder: ' + save_folder)

    training_dataset = DatasetFromCSV(TRAINING, TRAINING_DIR, shuffle=SHUFFLE, transforms=TRANSFORMS, dataset_file=training_dataset_file)
    training_dataloader = DataLoader(training_dataset, BATCH_SIZE, num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss(reduction='sum') #softmax and crossentropy
    optimizer = optim.SGD(net.parameters(), INITIAL_LEARNING_RATE)
    # optimizer = optim.Adam(net.parameters(), INITIAL_LEARNING_RATE)

    num_training_batchs = (training_dataset.data_len + BATCH_SIZE - 1)//BATCH_SIZE
    num_steps = (num_training_batchs + ACCUMULATE - 1)//ACCUMULATE
    step_size = BATCH_SIZE*ACCUMULATE
    last_step_size = (training_dataset.data_len - 1)%step_size + 1

    if INITIAL_MODEL == None:
        model_file = NETWORK + '_0.pth'
        torch.save(net.state_dict(), os.path.join(models_folder, model_file))
    save_steps_i = [i*num_steps//SAVES_PER_EPOCH for i in range(1, SAVES_PER_EPOCH + 1)]

    for epoch_i in range(1, EPOCHS + 1):
        str_buf = '\nEpoch ' + str(epoch_i) + ':'
        print(str_buf)
        if epoch_i == 1:
            str_buf = str_buf[1:]
        with open(results_file, 'a') as results:
            results.write(str_buf + '\n')
        with open(classification_error_file, 'a') as classification_error:
            classification_error.write(str_buf + '\n')
        with open(loss_log_file, 'a') as loss_log:
            loss_log.write(str_buf)

        str_buf2 = '\n\tLoss\t\tErrors' + step_size*'\t' + 'Elapsed Time\tStep\n'
        # print(str_buf2)
        with open(training_log_file, 'a') as training_log:
            training_log.write(str_buf + '\n' + str_buf2 + '\n')

        epoch_steps_elapsed = 0.0
        gt, c = [], []
        step_loss = 0.0
        save_i = 1
        step_i = 1
        step_begin = time.time()
        for batch_i, batch in enumerate(training_dataloader, 1):
            classification = net(batch[0].to('cuda:0'))
            # print("CLASSIFICATION", classification)
            loss = criterion(classification, batch[1].to('cuda:0'))
            # print("LOSS", loss)
            loss.backward()

            gt += batch[1].tolist()
            c += torch.max(classification, 1)[1].tolist()
            step_loss += loss.item()

            if batch_i%ACCUMULATE == 0 or batch_i == num_training_batchs:
                current_step_size = last_step_size if batch_i == num_training_batchs else step_size

                optimizer.step()
                optimizer.zero_grad()

                step_loss /= current_step_size
                step_elapsed = time.time() - step_begin
                epoch_steps_elapsed += step_elapsed

                str_buf = '\t{:.9f}'.format(step_loss)
                str_buf2 = '\n\tBatch_Loss = {:.9f}'.format(step_loss)
                print(str_buf2)
                with open(loss_log_file, 'a') as loss_log:
                    loss_log.write(str_buf2)

                for j in range(len(gt)):
                    str_buf += '\t'
                    if gt[j] != c[j]:
                        str_buf += str(gt[j]) + '->' + str(c[j])
                str_buf += '\t{:.3f}s'.format(step_elapsed)
                str_buf2 ='\n\tElapsed Time = {:.3f}s'.format(step_elapsed)
                print(str_buf2)
                percentage = str(10000*batch_i//num_training_batchs)
                while len(percentage) < 3:
                    percentage = '0' + percentage
                percentage = percentage[:-2] + '.' + percentage[-2:]
                str_buf += '\t\t' + str(step_i) + '/' + str(num_steps) + ' (' + percentage + '%)'
                str_buf2 = '\n\tStep = ' + str(step_i) + '/' + str(num_steps) + ' (' + percentage + '%)'
                print(str_buf2)
                with open(loss_log_file, 'a') as loss_log:
                    loss_log.write(str_buf2)
                with open(training_log_file, 'a') as training_log:
                    training_log.write(str_buf + '\n')

                if step_i in save_steps_i:
                    model_file = NETWORK + '_' + str(epoch_i) + '_' + str(save_i) + '.pth'
                    torch.save(net.state_dict(), os.path.join(models_folder, model_file))
                    save_i += 1
                    if TEST != None:
                        str_buf = '\n' + model_file + ' tests:'
                        print(str_buf)
                        with open(results_file, 'a') as results:
                            results.write(str_buf + '\n')
                        with open(classification_error_file, 'a') as classification_error:
                            classification_error.write(str_buf + '\n')
                        for csv_file, datasets_per_label, dataloaders_per_label in tests:
                            test(net, csv_file, datasets_per_label, dataloaders_per_label, results_file, classification_error_file)
                        print()

                if step_i == num_steps:
                    str_buf = '\tEpoch Steps Elapsed Time: {:.3f}s'.format(epoch_steps_elapsed)
                    print('\n\tSave folder: ' + save_folder)
                    print(str_buf)
                    with open(training_log_file, 'a') as training_log:
                        training_log.write(str_buf + '\n')
                else:
                    gt, c = [], []
                    step_loss = 0.0
                    step_i += 1
                    step_begin = time.time()

        if (epoch_i < LAST_EPOCH_FOR_LEARNING_RATE_DECAY) and (epoch_i%DECAY_STEP_SIZE == 0):
            for g in optimizer.param_groups:
                g['lr'] /= DECAY_RATE


        # if (epoch_i == EPOCHS):
        for g in optimizer.param_groups:
            print("\nLEARNING_RATE = {:.20f}".format(g['lr']))
            with open(loss_log_file, 'a') as loss_log:
                    loss_log.write("\nLEARNING_RATE = {:.20f}".format(g['lr']))
            with open(training_log_file, 'a') as training_log:
                    training_log.write("\nLEARNING_RATE = {:.20f}".format(g['lr']))


if __name__ == "__main__":
    main()
