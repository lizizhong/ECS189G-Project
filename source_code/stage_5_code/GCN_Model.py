# encoding = utf-8
import sys
sys.path.append('/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

from source_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from source_code.stage_5_code.Dataset_Loader import Dataset_Loader
# from source_code.stage_5_code.GCN_layers import GraphConvolution, GCN

from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# file_dir, filename = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_5_data/', 'pubmed'
# GraphDataset = Dataset_Loader()
# GraphDataset.dataset_source_folder_path = file_dir + filename
# GraphDataset.dataset_name = filename
# data_dict = GraphDataset.load()
    
# features, labels = data_dict['graph']['X'], data_dict['graph']['y']
# model = GCN(features.shape[1], 16, 3, 0.5)
# model.to(device)


def train(model, data_dict):
    
    model.to(device)
    # train_mask, y_train = data_dict['mask']['train_mask'], data_dict['labels']['y_train']
    adj, features, labels = data_dict['graph']['utility']['A'], data_dict['graph']['X'], data_dict['graph']['y']
    idx_train, idx_val = data_dict['train_test_val']['idx_train'], data_dict['train_test_val']['idx_val']
    idx_test = data_dict['train_test_val']['idx_test']
    
    writer = SummaryWriter()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 200
    global_step = 0
    accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

    for epoch in range(num_epochs):
        model.train()
        adj, features, labels = adj.to(device), features.to(device), labels.to(device)
        outputs = model(features, adj)
        # train_loss = get_loss(outputs, y_train, train_mask)
        # train_loss = F.nll_loss(outputs[train_mask], y_train)
        train_loss = F.nll_loss(outputs[idx_train], labels[idx_train])
        writer.add_scalar("train_loss/train_step", train_loss, global_step)
        global_step += 1
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        outputs = model(features, adj)

        val_loss = F.nll_loss(outputs[idx_test], labels[idx_test])
        print('Epoch:', epoch, 'train_loss:', train_loss.item(), 'val_loss:', val_loss.item())

    accuracy_evaluator.data = {'true_y': labels[idx_train].cpu().numpy(), 'pred_y': outputs[idx_train].max(1)[1].cpu().numpy()}
    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    writer.close()
    # test(data_dict)
    
    
def test(model, data_dict):
    model.to(device)
    accuracy_evaluator = Evaluate_Accuracy('test evaluator', '')
    # do the testing, and result the result
    adj, features, labels = data_dict['graph']['utility']['A'], data_dict['graph']['X'], data_dict['graph']['y']
    idx_test = data_dict['train_test_val']['idx_test']
    model.eval()
    with torch.no_grad():
        adj, features, labels = adj.to(device), features.to(device), labels.to(device)
        outputs = model(features, adj)
        
    accuracy_evaluator.data = {'true_y': labels[idx_test].cpu().numpy(), 'pred_y': outputs[idx_test].max(1)[1].cpu().numpy()}
    # print('Accuracy:', accuracy_evaluator.evaluate())
    
    return labels[idx_test].cpu().numpy(), outputs[idx_test].max(1)[1].cpu().numpy()
    

def run(model, data_dict):
    print('method running...')
    print('--start training...')
    train(model, data_dict)
    print('--start testing...')
    label, pred_y = test(model, data_dict)
    
    return {'pred_y': pred_y, 'true_y': label}


        
if __name__ == '__main__':
    
    train(data_dict)
        