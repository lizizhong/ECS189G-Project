'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
from source_code.base_class.method import method
from source_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 700
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 3e-4

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # self.fc_layer_0 = nn.Linear(784, 196)
        # self.activation_func_0 = nn.ReLU()
        self.fc_layer_1 = nn.Linear(784, 196)
        # self.fc_layer_1 = nn.Linear(784, 49)
        # check here for nn.ReLU doc: https://·pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(196, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.Softmax(dim=1)

        # self.drouput = nn.Dropout(0.1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        # h0 = self.activation_func_0(self.fc_layer_0(x))
        h1 = self.activation_func_1(self.fc_layer_1(x))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h1))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
        # scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
        writer = SummaryWriter()
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
            writer.add_scalar("Loss/train", train_loss, epoch)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                train_metric_dict, metric_report = accuracy_evaluator.evaluate()
                print('Epoch:', epoch, 'Accuracy:', train_metric_dict['Accuracy'][0],
                      'Precision', str(train_metric_dict['Precision'][0]) + ' +/- ' + str(train_metric_dict['Precision'][1]),
                      'Recall', str(train_metric_dict['Recall'][0]) + ' +/- ' + str(train_metric_dict['Recall'][1]),
                      'F1', str(train_metric_dict['F1'][0]) + ' +/- ' + str(train_metric_dict['F1'][1]),
                      'Loss:', train_loss.item())

        accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
        train_metric_dict, metric_report = accuracy_evaluator.evaluate()
        writer.close()

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

