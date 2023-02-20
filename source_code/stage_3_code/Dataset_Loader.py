'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
import torch
import matplotlib.pyplot as plt
from source_code.base_class.dataset import dataset
from torch.utils.data import Dataset, DataLoader


class Dataset_Loader(Dataset):

    def __init__(self, dataset_source_folder_path = None, dataset_source_file_name = None,
                 dName=None, dDescription=None, is_train=True):
        # super().__init__(dName, dDescription, is_train)
        self.dataset_source_folder_path = dataset_source_folder_path
        self.dataset_source_file_name = dataset_source_file_name
        self.dName, self.dDescription = dName, dDescription
        self.is_train = is_train
        self.loaded_data = self.load()
    
    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        print('training set size:', len(data['train']), 'testing set size:', len(data['test']))
        X_train, y_train = [item['image'] for item in data['train']], [item['label'] for item in data['train']]
        X_test, y_test = [item['image'] for item in data['test']], [item['label'] for item in data['test']]

        if self.is_train:
            return {'X_train': X_train, 'y_train': y_train}
        else:
            return {'X_test': X_test, 'y_test': y_test}

    def __len__(self):
        if self.is_train:
            return len(self.loaded_data['y_train'])
        else:
            return len(self.loaded_data['y_test'])

    def __getitem__(self, index):
        if self.is_train:
            img_feature = self.loaded_data['X_train'][index]
            img_label = self.loaded_data['y_train'][index]
        else:
            img_feature = self.loaded_data['X_test'][index]
            img_label = self.loaded_data['y_test'][index]

        if self.dName == 'stage 3 ORL training dataset' or \
                self.dName == 'stage 3 ORL test dataset':
            img_label -= 1

        return torch.FloatTensor(img_feature), img_label

    def show_pic(self, type='train'):
        for pair in self.data[type]:
            plt.imshow(pair['image'], cmap='Greys')
            plt.show()
            print(pair['label'])

if __name__ == '__main__':
    dataset_folder_path, file_name = '../../data/stage_3_data/', 'MNIST'
    data_split = True
    # load dataset
    train_data_obj = Dataset_Loader(is_train=True, dName='stage 3 MNIST training dataset',
                                    dDescription='MNIST dataset for project stage 3',
                                    dataset_source_folder_path=dataset_folder_path, dataset_source_file_name=file_name)

    train_dataloader = DataLoader(train_data_obj, shuffle=True, batch_size=1000)

    for index, data in enumerate(train_dataloader):
        # train_data_obj.__getitem__(index)
        print(index)
