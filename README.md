# ECS189G Project
## Stage 2
### Task Description
In stage 2, we aim to train an MLP (Multilayer Perceptron) model to do the data classification task based on PyTorch.

### Data
The dataset files we used in this stage consist of train.csv and test.csv, which are the pre-partitioned training dataset and test dataset. Each line in these datasets is an example, the training dataset has 60,000 examples and the test dataset has 10,000 examples for classification.

The training and test data files of stage 2 are in `data/stage_2_data`. You can use `source_code/Dataset_Loader.py` to load it. A parameter `data_split` is added in the `source_code/base_class_setting.py` and since we do not need to divide the dataset it is set to *False* in this stage.

(Since the size of training dataset exceeds GitHub's file size limit, so I just upload test dataset here. You need to add `train.csv` here to train the model.)
### Training
All of the related source code files are in `source_code/stage_2_code/`, and the training script is in `script/stage_2_script/script_mlp.py`. Use command `python script/stage_2_script/script_mlp.py` to train and test the MLP model.

In this stage, I have not used the configuration file, so if you want to change the training setting or model setting, you need to change the related parameter in `source_code/stage_2_code/Method_MLP.py`, and change the result file name in `script/stage_2_script/script_mlp.py`.

### Experiment Result
All of the testing experiment result files are in `result/stage_2_result`. Each file is named as *MLP_model + type of setting (model/training) + setting no. + _test* (same as in the report). 

The `.pkl` file saves the corresponding model's predict label and the true label of each test examples, and the `.txt` file saves the corresponding metrics result of test dataset.

In this stage, we use accuracy, precision, recall, and F1 score as our evaluation metric.

## Stage 3
### Task Description
In stage 3, we aim to train three CNN model to do the image classification task based on PyTorch.

### Data
The dataset files we used in this stage consist of minist dataset for handwritten digits, orl dataset for face images and cifar dataset for different kinds of color images. The training and test data files of stage 3 are loaded from `data/stage_3_data` using the code `source_code/Dataset_Loader.py`. *Note: I made a slightly change in the dataset class to make it easier to divide the batches and to disrupt the order of the training set batches, and I did not use the data from torchvision.*

(Since the size of training dataset exceeds GitHub's file size limit, so I just upload the ORL dataset. You need to download and add another two datasets here to train the model.)
### Training
All of the related source code files are in `source_code/stage_3_code/`, and the training script is in `script/stage_3_script/script_mnist__cnn.py`, `script/stage_3_script/script_orl__cnn.py` and `script/stage_3_script/script_cifar__cnn.py`. Use command like `python script/stage_3_script/script_cifar__cnn.py` to train and test these CNN models.

In this stage, I have not used the configuration file, so if you want to change the training setting or model setting, you need to change the related parameter in the model source code (i.e., `source_code/stage_3_code/ORL_CNN.py`), and change the corresponding result file name in the script files.

### Experiment Result
All of the testing experiment result files are in `result/stage_3_result`. 

The `.pkl` file saves the corresponding model's predict label and the true label of each test examples, and the `.txt` file saves the corresponding metrics result of test dataset.

In this stage, we use accuracy, precision, recall, and F1 score as our evaluation metric.

## Stage 4

### Task Description

In stage 4, we aim to train RNN/LSTM/GRU model to do the text classification and text generation task based on PyTorch.

### Data

The training and test data files of stage 3 are loaded from `data/stage_4_data` using the code `source_code/stage_4_code/Dataset_Loader.py`  for text classification task and `source_code/stage_4_code/Dataset_Loader_Generation.py`  for text generation task.

(Since the size of training dataset exceeds GitHub's file size limit, you need to download and put these two datasets at `data/stage_4_data` to train the model).

### Training

All of the related source code files are in `source_code/stage_4_code/`, and the training script is in `script/stage_4_script/script_rnn.py` and `script/stage_4_script/script_rnn_generation.py`. Use command like `python script/stage_4_script/script_rnn.py` to train and test models.

In this stage, I have not used the configuration file, so if you want to change the training setting or model setting, you need to change the related parameter in the model source code (i.e., `source_code/stage_4_code/Classification_RNN.py`), and change the corresponding result file name in the script files.

### Experiment Result

All of the testing experiment result files are in `result/stage_4_result`. 

The `.pkl` file saves the corresponding model's predict label and the true label of each test examples, the `.txt` file saves the corresponding metrics result of test dataset and the `.file` is the training log for the corresponding model. (For text generation task's files, the filename may not correct due to the unreasonable naming convention.)

For text generation task, I also write the predict function in `source_code/stage_4_code/Generation_RNN.py`,  you can use the corresponding script to train a model and customize your sentence beginning, then run this script to generate the text.

## Environment
```
python                    3.8.16
pytorch                   1.13.1
numpy                     1.23.5
scikit-learn              1.2.0
tensorboard               2.11.2
transformers              4.26.1
nltk			          3.8.1
```