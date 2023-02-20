from source_code.base_class.method import method
from source_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np

class CNN_MNIST(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 3
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    # def __init__(self, mName, mDescription):
    #     method.__init__(self, mName, mDescription)
    #     nn.Module.__init__(self)
    #
    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride= (1, 1))
    #     self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride= (1, 1))
    #     self.relu = nn.ReLU()
    #     self.dropout1 = nn.Dropout(0.25)
    #     self.dropout2 = nn.Dropout(0.5)
    #
    #     self.fc1 = nn.Linear(9216, 128)
    #     self.fc2 = nn.Linear(128, 10)
    #     self.softmax = nn.Softmax(dim=1)
    #
    # def forward(self, x):
    #     x = self.relu(self.conv1(x))
    #     x = self.relu(self.conv2(x))
    #     x = self.dropout1(F.max_pool2d(x, 2))
    #     x = torch.flatten(x, 1)
    #     x = self.fc2(self.dropout2(self.relu(self.fc1(x))))
    #     output = self.softmax(x)
    #
    #     return output

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2) # (16,14,14)
                     )
        self.conv2 = nn.Sequential( # (16,14,14)
                             nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14)
                             nn.ReLU(),
                             nn.MaxPool2d(2) # (32,7,7)
                             )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # (batch, 32,7,7) -> (batch, 32*7*7)
        output = self.out(x)
        return output

    def run(self, data):
        print('method running...')
        print('--start training...')
        self.train(data['train'])
        print('--start testing...')
        pred_y, label = self.test(data['test'])
        return {'pred_y': pred_y, 'true_y': label}

    def train(self, dataloader):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html

        writer = SummaryWriter()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            for index, data in enumerate(dataloader):
                img_feature, img_label = data
                img_feature = img_feature.unsqueeze(dim=1)
                y_pred = self.forward(img_feature)
                # convert y to torch.tensor as well
                y_true = torch.LongTensor(img_label)
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
            # scheduler.step()

                if index % 200 == 0:
                    accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

        accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
        print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

        writer.close()

    def test(self, dataloader):
        # do the testing, and result the result
        y_pred, y_true = [], []
        for index, data in enumerate(dataloader):
            img_feature, img_label = data
            img_feature = img_feature.unsqueeze(dim=1)
            batch_y_pred = self.forward(img_feature)
            # convert the probability distributions to the corresponding labels
            # instances will get the labels corresponding to the largest probability
            y_pred.extend(np.array(batch_y_pred.max(1)[1]))
            y_true.extend(np.array(img_label))

        return y_pred, y_true






