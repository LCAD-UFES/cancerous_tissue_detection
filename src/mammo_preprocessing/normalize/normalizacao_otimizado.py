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
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
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

BATCH_SIZE, ACCUMULATE = 32, 1

NUM_WORKERS = 2

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

    arq = open('dataset_mean_std_otim.txt', 'a')

    for i, image in enumerate(tqdm(training_dataset)):
        # if i >= 50:
        #     break
        dataset.append(image)
        
    dataset = torch.cat(dataset).to(dtype=torch.float32)
    
    ##mean image e std image
    std = dataset.std(dim=0)
    save_image(std, "std.png")
    texto = ("\nDataset Standard Deviation Image: " + str(std.shape))
    print(texto)
    arq.write(texto)

    mean = dataset.mean(dim=0)
    save_image(mean, "mean.png")
    texto = ("\nDataset Mean Image: "  + str(mean.shape))
    print(texto)
    arq.write(texto) 

    var = dataset.var(dim=0)
    save_image(var, "var.png")
    texto = ("\nDataset Variance Image: "  + str(var.shape))
    print(texto)
    arq.write(texto) 
    

    ##mean pixel e std pixel
    std = dataset.std()
    texto = ("\nDataset Standard Deviation Pixel: "  + str(std))
    print(texto)
    arq.write(texto)

    mean = dataset.mean()
    texto = ("\nDataset Mean Pixel: "  + str(mean))
    print(texto)
    arq.write(texto)

    var = dataset.var()
    texto = ("\nDataset Variance Pixel: "  + str(var))
    print(texto)
    arq.write(texto)

    arq.close()   


if __name__ == '__main__':
    main() 
