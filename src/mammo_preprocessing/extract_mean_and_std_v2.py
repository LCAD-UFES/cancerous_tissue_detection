import torch
from torchvision import datasets, transforms as T
import os, sys, argparse

def extract_mean_and_std(root):
    transform = T.Compose([T.ToTensor()])
    dataset = datasets.ImageFolder(root, transform=transform)

    means = []
    stds = []
    i = 0
    for img, target in dataset:
        means.append(torch.mean(img))
        stds.append(torch.std(img))
        i+=1

    mean = torch.mean(torch.tensor(means))
    std = torch.mean(torch.tensor(stds))

    print('mean: ', mean)
    print('std: ', std)



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