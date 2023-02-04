'''
Base SettingModule class for all experiment settings
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc

#-----------------------------------------------------
class setting:
    '''
    SettingModule: Abstract Class
    Entries: 
    '''
    
    setting_name = None
    setting_description = None
    
    dataset = None
    method = None
    result = None
    evaluate = None

    def __init__(self, sName=None, sDescription=None,sDataSplit=False):
        self.setting_name = sName
        self.setting_description = sDescription
        self.data_split = sDataSplit
    
    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        if self.data_split:
            self.train_dataset, self.test_dataset = sDataset[0], sDataset[1]
        else:
            self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        if self.data_split:
            print('train_dataset:', self.train_dataset.dataset_name, ', test_dataset:', self.test_dataset.dataset_name,
                  ', method:', self.method.method_name, ', setting:', self.setting_name, ', result:', self.result.result_name,
                  ', evaluation:', self.evaluate.evaluate_name)
        else:
            print('dataset:', self.dataset.dataset_name, ', method:', self.method.method_name,
                ', setting:', self.setting_name, ', result:', self.result.result_name, ', evaluation:', self.evaluate.evaluate_name)

    @abc.abstractmethod
    def load_run_save_evaluate(self):
        return
