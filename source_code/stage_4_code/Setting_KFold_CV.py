'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from source_code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader
from source_code.stage_4_code.Generation_RNN import run2
from source_code.stage_4_code.Classification_RNN import run1

class Setting_KFold_CV(setting):
    fold = 3
    batch_size = 3000
    
    def load_run_save_evaluate(self):
        
        # load dataset
        if self.data_split:
            # train_loaded_data = self.train_dataset.load()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
            # test_loaded_data = self.test_dataset.load()
        else:
            train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        if self.data_split:
            data = {'train': train_dataloader, 'test': test_dataloader}
            learned_result = run1(data=data, model=self.method)
            # save raw ResultModule
            self.result.data = learned_result
            # self.result.fold_count = fold_count
            self.result.save()

            self.evaluate.data = learned_result
            score_dict, metric_report = self.evaluate.evaluate()

            # return np.mean(score_list), np.std(score_list)
            return score_dict, metric_report
        else:
            data = {'train': train_dataloader, 'test': ["Who is she", "I am a", "Do you have", "What do you", "Time flies like", "How does a"]}
            learned_result = run2(data=data, model=self.method, dataset=self.dataset)
            return learned_result

        
        