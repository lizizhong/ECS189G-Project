import sys
sys.path.append('/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from source_code.stage_5_code.Dataset_Loader import Dataset_Loader
from source_code.stage_5_code.Result_Saver import Result_Saver
from source_code.stage_5_code.Setting_KFold_CV import Setting_KFold_CV
from source_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from source_code.stage_5_code.GCN_layers import GCN
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(0)
    torch.manual_seed(0)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    file_dir, filename = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_5_data/', 'cora'
    data_split = False
    # load dataset
    data_obj = Dataset_Loader()
    data_obj.dataset_source_folder_path = file_dir + filename
    data_obj.dataset_name = filename
    print('load {} dataset finished'.format(filename))
    
    data_dict = data_obj.load()
    features, labels = data_dict['graph']['X'], data_dict['graph']['y']

    method_obj = GCN(features.shape[1], 16, labels.max().item() + 1, 0.5)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/result/stage_5_result/Cora_'
    result_obj.result_destination_file_name = 'noweightdecay'

    setting_obj = Setting_KFold_CV('k fold cross validation', '', data_split)

    evaluate_obj = Evaluate_Accuracy('Four evaluate metrics: Accuracy & Precision & Recall & F1 Score')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_dict, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    score_dict, metric_report = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('GCN Accuracy: ' + str(score_dict['Accuracy'][0]) + ' +/- ' + str(score_dict['Accuracy'][1]))
    print('GCN Precision: ' + str(score_dict['Precision'][0]) + '+/-' + str(score_dict['Precision'][1]))
    print('GCN Recall: ' + str(score_dict['Recall'][0]) + '+/-' + str(score_dict['Recall'][1]))
    print('GCN F1 Score:' + str(str(score_dict['F1'][0])) + '+/-' + str(score_dict['F1'][1]))
    print('************ Finish ************')
    with open(result_obj.result_destination_folder_path + result_obj.result_destination_file_name + '.txt', 'a') as writer:
        writer.write(metric_report)
    writer.close()
    # ------------------------------------------------------


