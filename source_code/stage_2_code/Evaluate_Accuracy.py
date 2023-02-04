'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import numpy as np

from sklearn import metrics
from source_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    description = None
    
    def evaluate(self):
        print('evaluating performance...')
        acc = accuracy_score(self.data['true_y'], self.data['pred_y'])
        pre = precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        recall = recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        f1 = f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        evaluate_dict = {'Accuracy': [np.mean(acc), np.std(acc)],
                         'Precision': [np.mean(pre), np.std(pre)],
                         'Recall': [np.mean(recall), np.std(recall)],
                         'F1': [np.mean(f1), np.std(f1)]}
        report = metrics.classification_report(self.data['true_y'], self.data['pred_y'], digits=5)

        # print report
        print(metrics.confusion_matrix(self.data['true_y'], self.data['pred_y']))
        print(report)

        return evaluate_dict, report
        