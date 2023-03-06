import sys
sys.path.append('/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from source_code.stage_4_code.Dataset_Loader import Dataset_Loader
from source_code.stage_4_code.Classification_RNN import RNN_CLASS
from source_code.stage_4_code.Result_Saver import Result_Saver
from source_code.stage_4_code.Setting_KFold_CV import Setting_KFold_CV
from source_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    train_dataset_folder_path, file_name = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_classification/train/**/**.txt', ''
    test_dataset_folder_path = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_classification/test/**/**.txt'
    data_split = True
    # load dataset
    train_data_obj = Dataset_Loader(is_train=True, dName='stage 4 text classification training dataset',
                                    dDescription='text classification dataset for project stage 4',
                                    dataset_source_folder_path=train_dataset_folder_path, dataset_source_file_name=file_name)
    test_data_obj = Dataset_Loader(is_train=False, dName='stage 4 text classification test dataset',
                                    dDescription='text classification dataset for project stage 4',
                                   dataset_source_folder_path=test_dataset_folder_path, dataset_source_file_name=file_name)
    print('load dataset finished')

    method_obj = RNN_CLASS('RNN model for text classification', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/result/stage_4_result/BLSTM_'
    result_obj.result_destination_file_name = 'baseline'

    setting_obj = Setting_KFold_CV('k fold cross validation', '', data_split)

    evaluate_obj = Evaluate_Accuracy('Four evaluate metrics: Accuracy & Precision & Recall & F1 Score')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare([train_data_obj, test_data_obj], method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    score_dict, metric_report = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN Accuracy: ' + str(score_dict['Accuracy'][0]) + ' +/- ' + str(score_dict['Accuracy'][1]))
    print('RNN Precision: ' + str(score_dict['Precision'][0]) + '+/-' + str(score_dict['Precision'][1]))
    print('RNN Recall: ' + str(score_dict['Recall'][0]) + '+/-' + str(score_dict['Recall'][1]))
    print('RNN F1 Score:' + str(str(score_dict['F1'][0])) + '+/-' + str(score_dict['F1'][1]))
    print('************ Finish ************')
    with open(result_obj.result_destination_folder_path + result_obj.result_destination_file_name + '.txt', 'a') as writer:
        writer.write(metric_report)
    writer.close()
    # ------------------------------------------------------


