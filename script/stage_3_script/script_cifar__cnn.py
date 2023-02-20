from source_code.stage_3_code.Dataset_Loader import Dataset_Loader
from source_code.stage_3_code.CIFAR_CNN import CNN_CIFAR
from source_code.stage_3_code.Result_Saver import Result_Saver
from source_code.stage_3_code.Setting_KFold_CV import Setting_KFold_CV
from source_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    dataset_folder_path, file_name = '../../data/stage_3_data/', 'CIFAR'
    data_split = True
    # load dataset
    train_data_obj = Dataset_Loader(is_train=True, dName='stage 3 CIFAR training dataset',
                                    dDescription='CIFAR dataset for project stage 3',
                                    dataset_source_folder_path=dataset_folder_path, dataset_source_file_name=file_name)
    test_data_obj = Dataset_Loader(is_train=False, dName='stage 3 CIFAR test dataset',
                                    dDescription='CIFAR dataset for project stage 3',
                                   dataset_source_folder_path=dataset_folder_path, dataset_source_file_name=file_name)

    method_obj = CNN_CIFAR('CNN model for cifar classification', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/cifar_'
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
    print('CIFAR Accuracy: ' + str(score_dict['Accuracy'][0]) + ' +/- ' + str(score_dict['Accuracy'][1]))
    print('CIFAR Precision: ' + str(score_dict['Precision'][0]) + '+/-' + str(score_dict['Precision'][1]))
    print('CIFAR Recall: ' + str(score_dict['Recall'][0]) + '+/-' + str(score_dict['Recall'][1]))
    print('CIFAR F1 Score:' + str(str(score_dict['F1'][0])) + '+/-' + str(score_dict['F1'][1]))
    print('************ Finish ************')
    with open(result_obj.result_destination_folder_path + result_obj.result_destination_file_name + '.txt', 'a') as writer:
        writer.write(metric_report)
    writer.close()
    # ------------------------------------------------------


