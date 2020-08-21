##    EXECUTAR NA MONSTER 10.9.8.251
##    NECESSARIO ALOCAR TODA A BASE PARA CALCULAR O STD DA BASE
##    Valoares para automatic_cropped_dataset
##    mean = 0.0032
##    variance = 0.0032
##    std  = 0.056547386197822846
# len(training_dataloader) = 104568
# count_mean - esperado 104568 = 104568
# MEAN = tensor(0.0032)
# esperado = 5246803968
# elements = 5246803968
# variance = tensor(0.0032)
# Dataset Variance - tensor(0.0032)
# Dataset Standard Deviation-: 0.056547386197822846


from __future__ import division, print_function
import random  
import numpy as np
import pandas as pd
import cv2
import math
from datetime import datetime
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets



TRAINING = (
        '/home/lcad/sabrina/cbisddsm_train_2020_02_18.txt',
        # '/mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_OF10_automatic_cropped_dataset.txt', #20 imagens
        # '/mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/src/mammo_preprocessing/blackListnew.txt',
        # '/mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_train_2020_02_18.txt', 
)

TRAINING_DIR = (
        '/home/lcad/sabrina/automatic_cropped_dataset',
        # '/mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/automatic_cropped_dataset',
        # '/mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/automatic_cropped_with_black_images_dataset',
        # '/mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/bw_samples',
)

SHUFFLE = True

BATCH_SIZE, ACCUMULATE = 1, 1

NUM_WORKERS = 1

def calc_mean(training_dataloader):
    
    print("len(training_dataloader) = ", len(training_dataloader))

    mean = 0.0
    i,j,count, elements = 0,0,0,0
    arq = open('dataset_mean_std_confirm.txt', 'a')
    
    for data in training_dataloader:
        a, x, y = data.size()
        # x, y = data.size() #mnist
        count +=1

        data = data.type(torch.float32)
        data = data/255.0

        for i in range(x):
            for j in range(y):
                mean += data[0][i][j]
                # mean += data[i][j] #mnist
                elements += 1


    print("count_mean [esperado 55430] => ", count)

    print('iteractions [esperado 2781255680] => ', elements)
    
    texto = ("\nMean (mean / interactions) - : {:.20f} ".format(mean / elements))
    print(texto)
    arq.write(texto)

    texto = ("\nMean (mean / interactions) - : {:.4f} ".format(mean / elements))
    print(texto)
    arq.write(texto)

    arq.close()
    
    return (mean / elements)

def calc_variance(training_dataloader, total_mean):
    sum_var_mean = 0.0
    i, j, elements = 0,0,0

    arq = open('dataset_mean_std_confirm.txt', 'a')

    
    for data in training_dataloader:
        a, x, y = data.size()
        # x, y = data.size() #mnist

        data = data.type(torch.float32)
        data = data/255.0

        for i in range(x):
            for j in range(y):
                _mean = data[0][i][j] - total_mean
                # _mean = data[i][j] - total_mean #mnist
                sum_var_mean += (_mean * _mean)
                elements +=1
    
    print("iteractions = ", elements)

    print('sum_var_mean= ', sum_var_mean)


    texto = ("\nVariance - (sum_var_mean / iteractions) : {:.20f}".format(sum_var_mean / elements))
    print(texto)
    arq.write(texto)

    texto = ("\nVariance - (sum_var_mean / iteractions) : {:.4f}".format(sum_var_mean / elements))
    print(texto)
    arq.write(texto)
    arq.close()

    return (sum_var_mean / elements)


def calculate_std(var_mean):
    
    arq = open('dataset_mean_std_confirm.txt', 'a')

    std_mean = 0.0
    
    std_mean = math.sqrt(var_mean)
    
    texto = ("\nDataset Standard Deviation - (sqrt (variance)): {:.20f} ".format(std_mean))
    print(texto)
    arq.write(texto)

    texto = ("\nDataset Standard Deviation - (sqrt (variance)): {:.4f} ".format(std_mean))
    print(texto)
    arq.write(texto)

    arq.close()


class DatasetFromCSV(Dataset):
    def __init__(self, csv_files, root_dirs, label=None, shuffle=False): 
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


    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        # image = cv2.imread(self.images[i], 3) #color image
        image = cv2.imread(self.images[i], 0) # grayscale image
        return image


def main():
    training_dataset = DatasetFromCSV(TRAINING, TRAINING_DIR, shuffle=SHUFFLE)
    training_dataloader = DataLoader(training_dataset, BATCH_SIZE, num_workers=NUM_WORKERS)

    # from tqdm import tqdm
    # dataset = []
    # for i, image in tqdm(enumerate(training_dataset)):
    #     dataset.append(image)
    #     print(image.shape)
    #     if i > 50:
    #         break
    # dataset = np.array(dataset)
    # print(np.mean(dataset, axis=0).shape,np.std(dataset, axis=0).shape)
    # print(dataset.shape)
    # exit()


    #### USE MNIST DATASET TO CODE VALIDATION  ######
    # mnist_trainset = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=None))
    # mnist_trainset = mnist_trainset.dataset.data

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("\nInitiate Calc_Mean")
    print("Current Time =", current_time)
    mean = calc_mean(training_dataloader)
    # mean = calc_mean(mnist_trainset)
    print(mean)
    now = datetime.now() - now
    print('Initiate Calc_Variance')
    print("Time lapsed =", now)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print('\nInitiate Calc_Variance')
    print("Current Time =", current_time)
    variance = calc_variance(training_dataloader, mean)
    # variance = calc_variance(mnist_trainset, mean)
    print(variance)
    now = datetime.now() - now
    print('Initiate Calc_Variance')
    print("Time lapsed =", now)
    

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print('\nInitiate STD')
    print("Current Time =", current_time)
    calculate_std(variance)
    now = datetime.now() - now
    print('Initiate Calc_Variance')
    print("Time lapsed =", now)



if __name__ == '__main__':
    main() 

#Validation in Mnist
# Initiate Calc_Mean
# len(training_dataloader) =  60000
# count_mean -  =>  60000
# iteractions 47040000

# Mean (mean / interactions) - : 0.13113595545291900635 
# Mean (mean / interactions) - : 0.1311 
# tensor(0.1311)

# Initiate Calc_Variance
# iteractions =  47040000
# sum_var_mean=  tensor(3915670.7500)
# Variance - (sum_var_mean / iteractions) : 0.08324129879474639893
# Variance - (sum_var_mean / iteractions) : 0.0832
# tensor(0.0832)

# Initiate STD
# Dataset Standard Deviation - (sqrt (variance)): 0.28851568206034555741 
# Dataset Standard Deviation - (sqrt (variance)): 0.2885 