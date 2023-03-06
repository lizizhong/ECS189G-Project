import sys
sys.path.append('/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from source_code.base_class.method import method
from source_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from source_code.stage_4_code.Data_Loader_Generation import Generater_Dataset_Loader
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np

from transformers import BertTokenizer, BertModel
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained("prajjwal1/bert-mini")
bert_model.eval()

class GeneraterRNN(method, nn.Module):
    learning_rate = 1e-3
    max_epoch = 500
    
    # embedding_dim = 256
    input_size = 256
    hidden_size = 64
    
    model_type = 'gru'

    def __init__(self, mName, mDescription, data):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        n_vocab = data.n_words
        # self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim)
        
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, dropout=0.2)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, dropout=0.2)
        else:
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, dropout=0.2)
            
        self.fc = nn.Linear(self.hidden_size, n_vocab)
        
    def forward(self, x):
        # embedding = self.embedding(x)
        out, _ = self.rnn(x)
            
        output = self.fc(out)

        return output
    
    
def run2(model, data, dataset):
    print('method running...')
    print('--start training...')
    train(model, data['train'])
    print('--start testing...')
    gen_words = predict(model, dataset=dataset, texts=data['test'])
    return gen_words


def train(model, dataloader):
    # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html

    model.to(device)
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_function = nn.CrossEntropyLoss()
    global_step = 0
    # it will be an iterative gradient updating process
    # we don't do mini-batch, we use the whole input as one batch
    # you can try to split X and y into smaller-sized batches by yourself
    model.train()
    for epoch in range(model.max_epoch):  # you can do an early stop if self.max_epoch is too much...
        for index, data in enumerate(dataloader):
            previous_sen, following_sen = data
            previous_sen = previous_sen.to(device)
            following_sen = following_sen.to(device)
            y_pred = model(previous_sen)

            train_loss = loss_function(y_pred.transpose(1, 2), following_sen)
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
        # scheduler.step()

            if index % 200 == 0:
                print('Step:', global_step, 'Loss:', train_loss.item())
        
        if epoch % 10 == 0: 
            path_name = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/result/stage_4_result/model_gen/0304_lstm_model_' + str(epoch) + '.pt'
            torch.save(model.state_dict(), path_name)

    print('Epoch:', epoch, 'Loss:', train_loss.item())

    writer.close()
    
def predict(model, dataset, texts, next_words=30):
    model.to(device)
    model.eval()
    gen_list = []
    for text in texts:
        words = text.split(' ')
        # words = [w.lower() for w in words]
        
        with torch.no_grad():
            for i in range(next_words):
                indexed_tokens = tokenizer.convert_tokens_to_ids(words[i:])
                segments_ids = [1] * len(words[i:])
                    
                tokens_tensor, segments_tensors = torch.tensor([indexed_tokens]), torch.tensor([segments_ids])
                with torch.no_grad():
                    outputs = bert_model(tokens_tensor, segments_tensors)
                    hidden_states = outputs.last_hidden_state.squeeze(0)

                y_pred = model(torch.FloatTensor(hidden_states).unsqueeze(0).to(device))
                
                last_word_logits = y_pred[0][-1]
                p = torch.nn.functional.softmax(last_word_logits, dim=0).cpu().numpy()
                word_index = np.random.choice(len(last_word_logits), p=p)
                # word_index = np.argmax(p)
                words.append(dataset.index2word[word_index])
        gen_list.append(' '.join(words))

    return gen_list

if __name__ == '__main__':
    
    dataset_folder_path, file_name = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/', 'data'
    # load dataset
    train_data_obj = Generater_Dataset_Loader(is_train=True, dName='stage 4 text generation training dataset',
                                    dDescription='text generation dataset for project stage 4',
                                    dataset_source_folder_path=dataset_folder_path, dataset_source_file_name=file_name)
    print('load dataset finished')

    # method_obj = GeneraterRNN(mName='RNN', mDescription='RNN model for text generation', data=train_data_obj)
    model_path = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/result/stage_4_result/model_gen/0304_gru_model_260.pt'
    model = GeneraterRNN(mName='RNN', mDescription='RNN model for text generation', data=train_data_obj)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test = ["I don't understand", "Did I tell", "I always travel", "What do you", "Time flies like", "How does a", "I just surprised", "Every single morning", "Yesterday I bought", "Some people have", "What's the difference"]
    
    gen_list = predict(model=model, dataset=train_data_obj, texts=test)
    print(gen_list)
    
