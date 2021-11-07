import numpy as np
import glob
import random
#import cv2
import itertools
import pandas as pd
import numpy as np
import csv

import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
import torchvision.datasets as dset

class datasetTrain(Dataset):
    def __init__(self, folder_path):

        # Get list
        csv_path = folder_path + "/train_scaling_last.csv" 
        
        f = open(csv_path, 'r') # csv file read
        rdr = csv.reader(f)
        self.train_data = []
        self.train_target = []


        for line in rdr:
            if line[4] != '0': # num.of shop Remove
                line = [float(item) for item in line]
                self.train_data.append(line[0:4])
                self.train_target.append(line[5:])
        f.close()
        print('train data size :', len(self.train_data))
        print('train target size :', len(self.train_target))
        #self.train_data = [float(i) for i in self.train_data]
        #self.train_target = [float(i) for i in self.train_target]
        #print(self.train_target)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.train_data[index])
        target = torch.FloatTensor(self.train_target[index])

        return data, target*10

class datasetTest(Dataset):
    def __init__(self, folder_path):

        # Get list
        csv_path = folder_path + "/test_scaling_last.csv" 
        
        f = open(csv_path, 'r') # csv file read
        rdr = csv.reader(f)
        self.test_data = []
        self.test_target = []
        for line in rdr:
            if line[4] != '0': # num.of shop Remove
                line = [float(item) for item in line]
                self.test_data.append(line[0:4])
                self.test_target.append(line[5:])
        f.close()

        print('test data size :', len(self.test_data))
        print('test target size :', len(self.test_target))
     

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.test_data[index])
        target = torch.FloatTensor(self.test_target[index])

        return data, target*10











