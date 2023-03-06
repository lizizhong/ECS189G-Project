import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from source_code.base_class.method import method
from source_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN_CLASS(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 35
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    input_size = 256
    hidden_size = 64
    output_size = 2
    
    model_type = 'lstm'
    bidirectional = True

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=self.bidirectional)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=self.bidirectional)
        else:
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=self.bidirectional, dropout=0.3)
            
        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_size*4, self.output_size)
            # self.fc = nn.Linear(self.hidden_size*2, self.output_size)
        else:
            self.fc = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        
        out, _ = self.rnn(x)
        if self.bidirectional:
            output = torch.cat((out[:, 0, :], out[:, -1, :]), -1)
            # output = out[:, -1, :]
        else:
            # output = out[:, -1, :]
            output = torch.mean(out, dim=1)
            
        output = self.fc(output)

        return output
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    

def run1(model, data):
    print('method running...')
    print('--start training...')
    train(model, data['train'])
    print('--start testing...')
    pred_y, label = test(model, data['test'])
    return {'pred_y': pred_y, 'true_y': label}


def train(model, dataloader):
    # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html

    model.to(device)
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=0.0)
    scheduler = StepLR(optimizer, step_size=27, gamma=0.01)
    # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_function = nn.CrossEntropyLoss()
    # for training accuracy investigation purpose
    accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
    global_step = 0
    # it will be an iterative gradient updating process
    # we don't do mini-batch, we use the whole input as one batch
    # you can try to split X and y into smaller-sized batches by yourself
    model.train()
    for epoch in range(model.max_epoch):  # you can do an early stop if self.max_epoch is too much...
        for index, data in enumerate(dataloader):
            item_label, hidden_states = data
            hidden_states = hidden_states.to(device)
            y_pred = model(hidden_states)
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(item_label).to(device)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
            writer.add_scalar("train_loss/train_step", train_loss, global_step)
            global_step += 1

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if index % 200 == 0:
                accuracy_evaluator.data = {'true_y': y_true.cpu().numpy(), 'pred_y': y_pred.max(1)[1].cpu().numpy()}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
        
        if epoch % 50 == 0: 
            path_name = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/result/stage_4_result/model_cls/LSTM_model_' + str(epoch) + '.pt'
            torch.save(model.state_dict(), path_name)
        if scheduler:
                scheduler.step()

    accuracy_evaluator.data = {'true_y': y_true.cpu().numpy(), 'pred_y': y_pred.max(1)[1].cpu().numpy()}
    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    writer.close()

def test(model, dataloader):
    model.to(device)
    # do the testing, and result the result
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            item_label, hidden_states = data
            item_label = item_label.to(device)
            hidden_states = hidden_states.to(device)
            batch_y_pred = model(hidden_states)
            # convert the probability distributions to the corresponding labels
            # instances will get the labels corresponding to the largest probability
            y_pred.extend(np.array(batch_y_pred.max(1)[1].cpu().numpy()))
            y_true.extend(np.array(item_label.cpu().numpy()))

    return y_pred, y_true
