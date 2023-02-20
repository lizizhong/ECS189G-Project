'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from source_code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        if self.data_split:
            train_loaded_data = self.train_dataset.load()
            test_loaded_data = self.test_dataset.load()
        else:
            loaded_data = self.dataset.load()
        
        # kf = KFold(n_splits=self.fold, shuffle=True)
        
        # fold_count = 0
        score_dict = {}
        # for train_index, test_index in zip(kf.split(train_loaded_data['X']), kf.split(test_loaded_data['X'])):
        #     fold_count += 1
        #     print('************ Fold:', fold_count, '************')
        # no need to use cross-validator in this stage
        X_train, X_test = np.array(train_loaded_data['X']), np.array(test_loaded_data['X'])
        y_train, y_test = np.array(train_loaded_data['y']), np.array(test_loaded_data['y'])

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        # self.result.fold_count = fold_count
        self.result.save()

        self.evaluate.data = learned_result
        score_dict, metric_report = self.evaluate.evaluate()
        
        # return np.mean(score_list), np.std(score_list)
        return score_dict, metric_report
        