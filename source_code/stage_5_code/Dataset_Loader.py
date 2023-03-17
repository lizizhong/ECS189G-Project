
'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD
import sys
sys.path.append('/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template')
from source_code.base_class.dataset import dataset
import random
import torch
import numpy as np
import scipy.sparse as sp

# np.random.seed(0)

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels
    
    def label_index(self, labels):
        classes = set(labels)
        classes_index = {c: i for i, c in enumerate(classes)}
        label_index = np.array(list(map(classes_index.get, labels)), dtype=np.int32)
                               
        return label_index
    
    def sample_mask(self, idx, l):
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)
    
    def random_choose_label(self, label_index):
        idx_train, idx_val, idx_test = [], [], []
        
        if self.dataset_name == 'cora':
            N, num_class = 170, 7
            train_end_idx, val_end_idx = 20, 20
        elif self.dataset_name == 'citeseer':
            N, num_class = 240, 6
            train_end_idx, val_end_idx = 20, 40
        else:
            N, num_class = 240, 3
            train_end_idx, val_end_idx = 20, 40
        
        index_choose = [np.where(label_index == i) for i in range(num_class)]
        for i in range(num_class):
            label_sample = random.sample(index_choose[i][0].tolist(), N)
            idx_train.extend(label_sample[:train_end_idx])
            idx_val.extend(label_sample[train_end_idx:val_end_idx])
            idx_test.extend(label_sample[val_end_idx:])
            
        return idx_train, idx_val, idx_test
        
        
    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])
        

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))
        
        # idx to one-hot labels
        label_index = self.label_index(idx_features_labels[:, -1])

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        # labels = torch.LongTensor(onehot_labels)
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        # if self.dataset_name == 'cora':
        #     idx_train = range(140)
        #     idx_test = range(200, 1200)
        #     idx_val = range(1200, 1500)
        # elif self.dataset_name == 'citeseer':
        idx_train, idx_val, idx_test = self.random_choose_label(label_index)
        # elif self.dataset_name == 'pubmed':
        #     idx_train = range(60)
        #     idx_val = range(60, 1200)
        #     idx_test = range(6300, 7300)
        # #---- cora-small is a toy dataset I hand crafted for debugging purposes ---
        # elif self.dataset_name == 'cora-small':
        #     idx_train = range(5)
        #     idx_val = range(5, 10)
        #     idx_test = range(5, 10)

        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)
        idx_val = torch.LongTensor(idx_val)
        
        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}


if __name__ == '__main__':
    file_dir, filename = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_5_data/', 'citeseer'
    GraphDataset = Dataset_Loader()
    GraphDataset.dataset_source_folder_path = file_dir + filename
    GraphDataset.dataset_name = filename
    GraphDataset.load()