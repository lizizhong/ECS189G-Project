'''
Concrete IO class for a specific dataset
'''

import os
import sys
sys.path.append('~/code/ECS189G_Winter_2022_Source_Code_Template/source_code')

import nltk
import torch
from glob import glob, iglob
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from transformers import BertTokenizer, BertModel
from transformers import AutoModel

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = stopwords.words('english')
porter = PorterStemmer()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-tiny', output_hidden_states = True)
model = AutoModel.from_pretrained("prajjwal1/bert-mini")
model.eval()


class Generater_Dataset_Loader(Dataset):

    def __init__(self, dataset_source_folder_path = None, dataset_source_file_name = None,
                 dName=None, dDescription=None, is_train=True, max_length=4):
        # super().__init__(dName, dDescription, is_train)
        self.dataset_source_folder_path = dataset_source_folder_path
        self.dataset_source_file_name = dataset_source_file_name
        self.dName, self.dDescription = dName, dDescription
        self.is_train = is_train
        self.max_length = max_length
        
        self.words = self.load()
        self.uniq_words = self._get_uniq_words()
        self.n_words = 1
        
        self.index2word = {0: '[eos]'}
        self.index2word = self._load_index()
        self.word2index = {v:k for k, v in self.index2word.items()}
        self.word2indexes = [self.word2index[w] for w in self.words]

        # self.glove_embedding = self._load_glove_model()
        
        
    def load(self):
        filename = self.dataset_source_folder_path + self.dataset_source_file_name
        train_df = pd.read_csv(filename)
        text = train_df['Joke'].str.cat(sep=' ')

        # clean data
        item_words = word_tokenize(text)
        words = [word for word in item_words if word.isalpha()]
        # words = [w for w in words if not w in stop_words]
        # stemmed = [porter.stem(word) for word in words]
        # content = ' '.join(stemmed)
        return words
    
    def _load_index(self):
        for index, word in enumerate(self.uniq_words):
            self.index2word[self.n_words] = word
            self.n_words += 1
        
        return self.index2word
    
    def _addSentence(self, sentence):
        words = word_tokenize(sentence)
        for word in words:
            self._addWord(word)
    
    def _addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2index[word] += 1
            
    def _get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
            
    def _load_glove_model(self):
        
        glove_file = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_classification/glove.6B.300d.txt'
        glove_embedding = {}
        
        with open(glove_file, 'r', encoding='utf-8') as f:
            contents = f.readlines()
        f.close()
        
        for content in tqdm(contents[1:]):
            split_line = content.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_embedding[word] = embedding
        
        glove_embedding['[unk]'] = np.zeros(300)
        glove_embedding['[eos]'] = np.ones(300)
        
        return glove_embedding       
    
    def __len__(self):
        # return len(self.data_dict)
        return len(self.word2indexes) - self.max_length
    
    def __getitem__(self, index):
        
        pre_sen, fol_sen = self.words[index:index+self.max_length], self.word2indexes[index+1:index+self.max_length+1]
        pre_sen_embed, fol_sen_embed = [], []
        # for word in pre_sen:
        #     if word in self.glove_embedding:
        #         pre_sen_embed.append(self.glove_embedding[word])
        #     else:
        #         pre_sen_embed.append(self.glove_embedding['[unk]'])
        # marked_content = ' '.join(pre_sen)
        # tokenized_content = tokenizer.tokenize(marked_content)
        indexed_tokens = tokenizer.convert_tokens_to_ids(pre_sen)
        segments_ids = [1] * min(len(pre_sen), self.max_length)
        
        tokens_tensor, segments_tensors = torch.tensor([indexed_tokens]), torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs.last_hidden_state.squeeze(0)
                
        return torch.FloatTensor(hidden_states), torch.tensor(fol_sen)
        
    
if __name__ == '__main__':
    dataset_folder_path, file_name = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/', 'data'
    # load dataset
    train_data_obj = Generater_Dataset_Loader(is_train=True, dName='stage 4 text generation test dataset',
                                    dDescription='text generation dataset for project stage 4',
                                    dataset_source_folder_path=dataset_folder_path, dataset_source_file_name=file_name)
    
    train_data_obj.__getitem__(3)
    
    # train_dataloader = DataLoader(train_data_obj, batch_size=500, shuffle=False)
    
    # for index, data in enumerate(train_dataloader):
    #     print(index)