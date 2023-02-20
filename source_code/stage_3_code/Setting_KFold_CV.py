'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from source_code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        if self.data_split:
            # train_loaded_data = self.train_dataset.load()
            train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)
            test_dataloader = DataLoader(self.test_dataset, batch_size=100, shuffle=False)
            # test_loaded_data = self.test_dataset.load()
        else:
            loaded_data = self.dataset.load()

        data = {'train': train_dataloader, 'test': test_dataloader}
        learned_result = self.method.run(data)

        # save raw ResultModule
        self.result.data = learned_result
        # self.result.fold_count = fold_count
        self.result.save()

        self.evaluate.data = learned_result
        score_dict, metric_report = self.evaluate.evaluate()

        # return np.mean(score_list), np.std(score_list)
        return score_dict, metric_report
        