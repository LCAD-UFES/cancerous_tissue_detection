##    EXECUTAR NA MONSTER 10.9.8.251
##    NECESSARIO ALOCAR TODA A BASE PARA CALCULAR O STD DA BASE

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
from torchvision.transforms import ToTensor
from tqdm import tqdm

TRAINING = (
        '/home/lcad/sabrina/cbisddsm_train_2020_02_18.txt',
        # '/mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/src/squeezeNet/aux_files/cbisddsm_train_2020_02_18.txt', 
)

TRAINING_DIR = (
        '/home/lcad/sabrina/automatic_cropped_dataset',
        # '/mnt/dadosSabrina/MyDrive/breast_cancer_analyzer_LCAD/dataset/cancer_tissue_dataset/automatic_cropped_dataset',
)

SHUFFLE = False

BATCH_SIZE, ACCUMULATE = 1, 1

NUM_WORKERS = 8

def calc_mean(dataset):
    
    print("len(dataset) = ", len(dataset))

    mean = 0.0
    i,j,count, elements = 0,0,0,0
    arq = open('dataset_mean_std_confirm.txt', 'a')
    
    for data in dataset:
        a, x, y = data.size()
        count +=1

        data = data.type(torch.float32)
        data = data/255.0

        for i in range(x):
            for j in range(y):
                mean += data[0][i][j]
                elements += 1


    print("count_mean (quant images)  => ", count)

    print('iteractions => ', elements)
    
    texto = ("\nMean (total mean / interactions) - : {:.20f} ".format(mean / elements))
    print(texto)
    arq.write(texto)

    texto = ("\nMean (total mean / interactions) - : {:.4f} ".format(mean / elements))
    print(texto)
    arq.write(texto)

    arq.close()
    
    return (mean / elements)

def calc_variance(dataset, total_mean):
    sum_var_mean = 0.0
    i, j, elements = 0,0,0

    arq = open('dataset_mean_std_confirm.txt', 'a')

    
    for data in dataset:
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
    def __init__(self, csv_files, root_dirs, label=None, shuffle=False, transform=None): 
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
        self.transform = transform

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
        if self.transform:
            image = self.transform(image)
        return image


def main():
    training_dataset = DatasetFromCSV(TRAINING, TRAINING_DIR, shuffle=SHUFFLE, transform=ToTensor())    

    dataset = []
    
    for i, image in enumerate(tqdm(training_dataset)):
        # if i >= 50:
        #     break
        dataset.append(image)
        # print(image.shape)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("\nInitiate Calc_Mean")
    print("Current Time =", current_time)
    mean = calc_mean(dataset)
    # mean = calc_mean(mnist_trainset)
    print(mean)
    now = datetime.now() - now
    print('Initiate Calc_Variance')
    print("Time lapsed =", now)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print('\nInitiate Calc_Variance')
    print("Current Time =", current_time)
    variance = calc_variance(dataset, mean)
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