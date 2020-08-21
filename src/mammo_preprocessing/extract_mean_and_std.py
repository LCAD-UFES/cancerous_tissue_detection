import torch
from torchvision import datasets, transforms as T
import os, sys, argparse
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from numpy import asarray
import numpy as np 
import pandas as pd
scaler = MinMaxScaler() 

def extract_mean_and_std(root):
    it = 0
    for root, dirs, files in os.walk(root):
        for i, imgs in enumerate(files):
            # load the image
            image = Image.open(root + '/' + imgs)
            # convert image to numpy array
            
            data = asarray(image)
            print(type(data))

            # summarize shape

            print(data.shape)

            if it == 0:
                dataset = asarray(image)
            #if it > 0:
                #np.append(data, dataset)    
            print(i)
            it+=1
            if it == 1:
                print(data)
                break
                
    data_scaled = scaler.fit_transform(dataset)

    #print('means (Loan Amount, Int rate and Installment): ', data_scaled.mean(axis=0))
    #print('std (Loan Amount, Int rate and Installment): ', data_scaled.std(axis=0))




def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="the path of Dataset")
    args = parser.parse_args()
    root = args.p
    print(args.p)

    extract_mean_and_std(root)
    
    return 0
 

if __name__ == '__main__':
    sys.exit(main(sys.argv)) 
