'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import re
import sys
import json
import nltk
import string
sys.path.append('~/code/ECS189G_Winter_2022_Source_Code_Template/source_code')

import torch
from glob import glob, iglob
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = stopwords.words('english')
porter = PorterStemmer()

from transformers import BertTokenizer, BertModel
from transformers import AutoModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-tiny', output_hidden_states = True)
model = AutoModel.from_pretrained("prajjwal1/bert-mini")
model.eval()

class Dataset_Loader(Dataset):

    def __init__(self, dataset_source_folder_path = None, dataset_source_file_name = None,
                 dName=None, dDescription=None, is_train=True, max_length=128):
        # super().__init__(dName, dDescription, is_train)
        self.dataset_source_folder_path = dataset_source_folder_path
        self.dataset_source_file_name = dataset_source_file_name
        self.dName, self.dDescription = dName, dDescription
        self.is_train = is_train
        self.max_length = max_length
        self.table = str.maketrans('', '', string.punctuation)

        self.data_dict= self.load()
    
    
    def load(self):
        
        data_dict = {"label": [], "content": [], 'filename': []}
        filenames = glob(self.dataset_source_folder_path + self.dataset_source_file_name)
        for index, filename in enumerate(filenames):
            data_dict['filename'].append(filename)
            label_str = int(filename.split('/')[-1].split('.')[0].split('_')[1])
            label = 1 if label_str >= 7 else 0
            data_dict['label'].append(label)
            with open (filename, encoding='utf-8') as f:
                contents = f.readlines()
                assert len(contents) == 1
                for content in contents:
                    content = content.strip()
                data_dict['content'].append(content)
            f.close()
        
        return data_dict
    
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
        
        glove_embedding['[pad]'] = np.zeros(300)
        
        return glove_embedding
        
    
    def _load_embedding(self, item_content):
        """For bert embedding loading
        """
        # clean data
        item_words = word_tokenize(item_content)
        words = [word for word in item_words if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        stemmed = [porter.stem(word) for word in words]
        content = ' '.join(stemmed)
        
        marked_content = "[CLS] " + content + " [SEP]"
        tokenized_content = tokenizer.tokenize(marked_content)
        if len(tokenized_content) < self.max_length:
            tokenized_content = self._padding(tokenized_content)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_content) [:self.max_length]
        segments_ids = [1] * min(len(tokenized_content), self.max_length)
        
        tokens_tensor, segments_tensors = torch.tensor([indexed_tokens]), torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs.last_hidden_state.squeeze(0)
            # hidden_states = outputs[2]
            # token_embeddings = torch.stack(hidden_states, dim=0)
            # token_embeddings = torch.squeeze(token_embeddings, dim=1)
            # token_embeddings = token_embeddings.permute(1,0,2)
            # for token in token_embeddings:
            #     sum_vec = torch.sum(token[-4:], dim=0)
            #     token_vecs_sum.append(sum_vec)

        return hidden_states
                
    def __len__(self):
        return len(self.data_dict['label'])
    
    def _padding(self, tokenized_content):
        pad_word = '[PAD]'
        while len(tokenized_content) < self.max_length:
            tokenized_content.append(pad_word)
            
        return tokenized_content

    def __getitem__(self, index):
        
        item_label = self.data_dict['label'][index]
        
        content = self.data_dict['content'][index]
        content_embedding = self._load_embedding(content)
        
        return item_label, torch.tensor(content_embedding)
    
    
def convert_dict():
    readfile = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_classification/train_128_embedding.json'
    outfile = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_classification/new_train_128_embedding.json'
    with open(readfile, encoding='utf-8-sig') as f:
        data_dict = json.load(f)
        f.close()
    label, filename, content_embedding = data_dict['label'], data_dict['filename'], data_dict['content_embedding']
    new_dict = {}
    for index, item in tqdm(enumerate(label)):
        new_dict[content_embedding[index]]=item
    
    with open(outfile, 'w', encoding='utf-8-sig') as f:
        json.dump(new_dict, f, indent=2, sort_keys=True, ensure_ascii=False)
    

if __name__ == '__main__':
    dataset_folder_path, file_name = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_classification/test/**/**.txt', ''
    data_split = True
    # load dataset
    train_data_obj = Dataset_Loader(is_train=True, dName='stage 4 text classification test dataset',
                                    dDescription='text classification dataset for project stage 4',
                                    dataset_source_folder_path=dataset_folder_path, dataset_source_file_name=file_name)

    train_dataloader = DataLoader(train_data_obj, shuffle=True, batch_size=500)

    # for index, data in enumerate(train_dataloader):
    train_data_obj.__getitem__(3)
        # print(index)
