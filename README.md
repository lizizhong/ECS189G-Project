# ECS189G Project
## Stage 2
### Task Description
In stage 2, we aim to train an MLP (Multilayer Perceptron) model to do the data classification task based on PyTorch.

### Data
The dataset files we used in this stage consist of train.csv and test.csv, which are the pre-partitioned training dataset and test dataset. Each line in these datasets is an example, the training dataset has 60,000 examples and the test dataset has 10,000 examples for classification.

The training and test data files of stage 2 are in `data/stage_2_data`. You can use `source_code/Dataset_Loader.py` to load it. A parameter `data_split` is added in the `source_code/base_class_setting.py` and since we do not need to divide the dataset it is set to *False* in this stage.

(Since the size of training dataset exceeds GitHub's file size limit, so I just upload test dataset here. You need to add `train.csv` here to train the model.)
### Training
All of the related source code files are in `source_code/stage_2_code/', and the training script is in `script/stage_2_script/script_mlp.py`. Use command `python script/stage_2_script/script_mlp.py` to train and test the MLP model.

In this stage, I have not used the configuration file, so if you want to change the training setting or model setting, you need to change the related parameter in `source_code/stage_2_code/Method_MLP.py`, and change the result file name in `script/stage_2_script/script_mlp.py`.

### Experiment Result
All of the testing experiment result files are in `result/stage_2_result`. Each file is named as *MLP_model + type of setting (model/training) + setting no. + _test* (same as in the report). 

The `.pkl` file saves the corresponding model's predict label and the true label of each test examples, and the `.txt` file saves the corresponding metrics result of test dataset.

In this stage, we use accuracy, precision, recall, and F1 score as our evaluation metric.

### Environment
```
python                    3.8.16
pytorch                   1.13.1
numpy                     1.23.5
scikit-learn              1.2.0
tensorboard               2.11.2
```