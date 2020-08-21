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


RUNS_FOLDER = '/mnt/dadosSabrina/breast_cancer_analyzer_LCAD/src/squeezetnet/runs'

NETWORK = 'squeezenet1_1'
NUM_CLASSES = 2

INITIAL_MODEL = None
INITIAL_MODEL_TEST = False

TRAINING = (
        '/mnt/dadosSabrina/breast_cancer_analyzer_LCAD/src/squeezetnet/aux_files/cbisddsm_train_2019_10_15.txt',
)

TRAINING_DIR = (
        '/mnt/dadosSabrina/breast_cancer_analyzer_LCAD/dataset',
)

SHUFFLE = True

TEST = (
         '/mnt/dadosSabrina/breast_cancer_analyzer_LCAD/src/squeezetnet/aux_files/cbisddsm_val_2019_10_15.txt',
        # '/home/sabrina/dataset/cbisddsm_test_2019_09_12.txt',
)
TEST_DIR = (
         '/mnt/dadosSabrina/breast_cancer_analyzer_LCAD/dataset',
        # '/home/sabrina/dataset',
)

TRANSFORMS = transforms.Normalize([0.4818, 0.4818, 0.4818], [0.1752, 0.1752, 0.1752])

BATCH_SIZE, ACCUMULATE = 128, 1

EPOCHS = 100
SAVES_PER_EPOCH = 10

INITIAL_LEARNING_RATE = 0.0004
LAST_EPOCH_FOR_LEARNING_RATE_DECAY = 7
DECAY_RATE = 2
DECAY_STEP_SIZE = 2

NUM_WORKERS = 4


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
    load_matching_name_and_shape_layers(net, 'Torchvision pretrained model', model(pretrained=True).state_dict())
    return net

class DatasetFromCSV(Dataset):
    def __init__(self, csv_files, root_dirs, label=None, shuffle=False, transforms=None, dataset_file=None):
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

        self.transforms = transforms

        if dataset_file != None:
            with open(dataset_file, 'w') as dataset:
                for image, label in zip(self.images, self.labels):
                    dataset.write(image + ' ' + str(label) + '\n')

    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        image = cv2.imread(self.images[i], 3)
        image = np.transpose(image, [2, 0, 1])[[2, 1, 0]]
        image = image/255
        image = torch.from_numpy(image.astype(np.float32))
        if self.transforms != None:
            image = self.transforms(image)
        return (image, self.labels[i])


def test(net, dataset_name, datasets_per_label, dataloaders_per_label, results_file=None):
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
                    c = torch.max(classification, 1)[1].tolist()
                    for j in range(NUM_CLASSES):
                        line[j] += c.count(j)
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
    results_file = os.path.join(save_folder, 'results.txt')
    print('\nSave folder: ' + save_folder)

    training_dataset = DatasetFromCSV(TRAINING, TRAINING_DIR, shuffle=SHUFFLE, transforms=TRANSFORMS, dataset_file=training_dataset_file)
    training_dataloader = DataLoader(training_dataset, BATCH_SIZE, num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(net.parameters(), INITIAL_LEARNING_RATE)

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
        str_buf2 = '\n\tLoss\t\tErrors' + step_size*'\t' + 'Elapsed Time\tStep\n'
        print(str_buf2)
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
            loss = criterion(classification, batch[1].to('cuda:0'))
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
                for j in range(len(gt)):
                    str_buf += '\t'
                    if gt[j] != c[j]:
                        str_buf += str(gt[j]) + '->' + str(c[j])
                str_buf += '\t{:.3f}s'.format(step_elapsed)
                percentage = str(10000*batch_i//num_training_batchs)
                while len(percentage) < 3:
                    percentage = '0' + percentage
                percentage = percentage[:-2] + '.' + percentage[-2:]
                str_buf += '\t\t' + str(step_i) + '/' + str(num_steps) + ' (' + percentage + '%)'
                print(str_buf)
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
                        for csv_file, datasets_per_label, dataloaders_per_label in tests:
                            test(net, csv_file, datasets_per_label, dataloaders_per_label, results_file)
                        print()

                if step_i == num_steps:
                    str_buf = '\tEpoch Steps Elapsed Time: {:.3f}s'.format(epoch_steps_elapsed)
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


if __name__ == "__main__":
    main()
