from source_code.base_class.method import method
from source_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np

class CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 25
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(128 * 4 * 4, 512)
        # self.fc2 = nn.Linear(512, 10)
        #
        # x = torch.relu(self.conv1(x))
        # x = self.pool(torch.relu(self.conv2(x)))
        # x = self.pool(torch.relu(self.conv3(x)))
        # x = x.view(-1, 128 * 4 * 4)
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)

        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                     nn.ReLU()
                     )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
                             nn.Conv2d(64, 128, 3, 1, 1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2)
                             )
        self.classifier = nn.Sequential(
                    nn.Linear(128*8*8, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
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

        global_step = 0
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            for index, data in enumerate(dataloader):
                img_feature, img_label = data
                img_feature = img_feature.transpose(1, -1).contiguous()
                y_pred = self.forward(img_feature)
                # convert y to torch.tensor as well
                y_true = torch.LongTensor(img_label)
                # calculate the training loss
                train_loss = loss_function(y_pred, y_true)
                writer.add_scalar("train_loss/global_step", train_loss, global_step)
                writer.add_scalar("train_loss/epoch", train_loss, epoch)
                global_step += 1

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
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
            img_feature = img_feature.transpose(1, -1).contiguous()
            batch_y_pred = self.forward(img_feature)
            # convert the probability distributions to the corresponding labels
            # instances will get the labels corresponding to the largest probability
            y_pred.extend(np.array(batch_y_pred.max(1)[1]))
            y_true.extend(np.array(img_label))

        return y_pred, y_true






