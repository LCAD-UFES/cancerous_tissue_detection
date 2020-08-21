# python test_cancer_tissue.py runs_manual_cropped_dataset/squeezenet1_1/01/models/ runs_manual_cropped_dataset/squeezenet1_1/01/all_confusion_matrix_with_testSet.txt

from __future__ import division, print_function
import numpy as np

import sys
import os
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torchvision import models, transforms


from cancer_tissue_dataset import DatasetFromCSV
from read_yaml import Read_Yaml

NUM_CLASSES = 2
#SHUFFLE = True
TRANSFORMS = transforms.Normalize([0.4818, 0.4818, 0.4818], [0.1752, 0.1752, 0.1752])
NUM_WORKERS = 4

HYPERPARAMETERS = '/home/pedro/IC_LCAD/breast_cancer_analyzer_LCAD/src/squeezeNet/dicom/hyperparameters.yaml'
DIRS = '/home/pedro/IC_LCAD/breast_cancer_analyzer_LCAD/src/squeezeNet/dicom/dirs.yaml'

directories = Read_Yaml(DIRS)
NETWORK = directories.params.network
INITIAL_MODEL = directories.params.initial_model
INITIAL_MODEL_TEST = directories.params.initial_model_test
TEST = directories.params.test
TEST_DIR = directories.params.test_dir

hyperparameters = Read_Yaml(HYPERPARAMETERS)
BATCH_SIZE = hyperparameters.params.batch_size


def load_matching_name_and_shape_layers(net, new_model_name, new_state_dict):
    # print('\n' + new_model_name + ':')
    state_dict = net.state_dict()
    for key in state_dict:
        if key in new_state_dict and new_state_dict[key].shape == state_dict[key].shape:
            state_dict[key] = new_state_dict[key]
            # print('\t' + key + ' loaded.')
    net.load_state_dict(state_dict)

def Net():
    model = getattr(models, NETWORK)
    net = model(num_classes=NUM_CLASSES)
    # load_matching_name_and_shape_layers(net, 'Torchvision pretrained model', model(pretrained=True).state_dict())
    return net


def test(net, dataset_name, datasets_per_label, dataloaders_per_label, results_file):
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
                    m = nn.Softmax(dim=1)
                    batch_s = m(classification)
                    batch_s = batch_s.tolist()
                    c = torch.max(classification, 1)[1].tolist()
                    for j in range(NUM_CLASSES):
                        line[j] += c.count(j)
            class_accuracy = float(line[i])/dataset.data_len
            average_class_accuracy += class_accuracy
        str_buf = '\t'
        for j in range(NUM_CLASSES):
            str_buf += '\t' + str(line[j])
        str_buf += '\t{:.9f}'.format(class_accuracy)
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


def main(args):
    torch.multiprocessing.set_start_method('spawn', force=True)

    model_path = sorted(Path(os.path.join(args[1])).rglob('*.pth'))

    results_file = args[2]#'test_results_03.txt'

    for INITIAL_MODEL in model_path:
        INITIAL_MODEL = str(INITIAL_MODEL)

        torch.multiprocessing.set_start_method('spawn', force=True)

        net = Net().to('cuda:0')
        if INITIAL_MODEL != None:
            load_matching_name_and_shape_layers(net, INITIAL_MODEL, torch.load(INITIAL_MODEL))


        if TEST != None:
            if INITIAL_MODEL_TEST:
                print('\n' + (INITIAL_MODEL if INITIAL_MODEL != None else 'Initial model') + ' tests:')
                with open(results_file, 'a') as results:
                    results.write(INITIAL_MODEL + '\n')
            tests = []
            for csv_file, root_dir in zip(TEST, TEST_DIR):
                datasets_per_label = [DatasetFromCSV((csv_file,), (root_dir,), label=i, transforms=TRANSFORMS, dataset_file='test_dataset.txt') for i in range(NUM_CLASSES)]
                dataloaders_per_label = [DataLoader(dataset, BATCH_SIZE, num_workers=NUM_WORKERS) for dataset in datasets_per_label]
                tests.append((csv_file, datasets_per_label, dataloaders_per_label))
                if INITIAL_MODEL_TEST:
                    test(net, csv_file, datasets_per_label, dataloaders_per_label, results_file)


if __name__ == '__main__':
    sys.exit(main(sys.argv)) 



