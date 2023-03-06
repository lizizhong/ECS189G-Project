import sys
sys.path.append('/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from source_code.stage_4_code.Data_Loader_Generation import Generater_Dataset_Loader
from source_code.stage_4_code.Generation_RNN import GeneraterRNN
from source_code.stage_4_code.Result_Saver import Result_Saver
from source_code.stage_4_code.Setting_KFold_CV import Setting_KFold_CV
from source_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(0)
    torch.manual_seed(0)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    dataset_folder_path, file_name = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/text_generation/', 'data'
    data_split = False
    # load dataset
    train_data_obj = Generater_Dataset_Loader(is_train=True, dName='stage 4 text generation training dataset',
                                    dDescription='text generation dataset for project stage 4',
                                    dataset_source_folder_path=dataset_folder_path, dataset_source_file_name=file_name)
    print('load dataset finished')

    method_obj = GeneraterRNN(mName='RNN', mDescription='RNN model for text generation', data=train_data_obj)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '/home/zizhong/code/ECS189G_Winter_2022_Source_Code_Template/result/stage_4_result/gen_gru_'
    result_obj.result_destination_file_name = 'baseline'

    setting_obj = Setting_KFold_CV('k fold cross validation', '', data_split)

    evaluate_obj = Evaluate_Accuracy('Four evaluate metrics: Accuracy & Precision & Recall & F1 Score')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    gen_words = setting_obj.load_run_save_evaluate()
    print(gen_words)
    with open(result_obj.result_destination_folder_path + result_obj.result_destination_file_name + '.txt', 'a') as writer:
        for item in gen_words:
            writer.write(item)
            writer.write('\n')
    writer.close()
    # ------------------------------------------------------


