'''
## Author: Puja Chowdhury
## Email: pujac@email.sc.edu
## Date: 07/18/2023
Configuring the model
'''
import torch
## Checking GPU
def device_configuration():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
        print("Number of GPU {}".format(torch.cuda.device_count()))
        print("GPU type {}".format(torch.cuda.get_device_name(0)))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        # torch.cuda.memory_summary(device=None, abbreviated=False)
        # print(torch.backends.cudnn.version())
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    return device
# Setting device
device = device_configuration()

## setting the mode
train_teacher = True
train_student = True

## data and file processing
input_file = "../data/test3_ansys_data.csv"
## Sample based prediction
input_window = 25500 # Training time in samples, 25500
output_window = 500 # Prediction time in samples, 500
stride = 500 # Sliding time in samples, 500
num_features = 1 # 3
## Time based prediction. Avoid this as it will create problems for windowing dataset.
time_based = False
input_time=0.5 #training time
time_to_predict=0.010 # time of prediction
series_length=9
sliding_size= 0.009765605 # 0.001 0.009765605 #
computation_time=0

## Hyperparameters for teachers
encoder_size = 8 # Synthetic: 1; Real: 8
hidden_size = 32 # best: 1f: 32, 3f: 32
n_epochs = 50 # best 1f:100, 3f:50
batch_size = 8 # best: 8
learning_rate_teacher = 0.001 # best: 0.001 (best for 1 features), 3f: 0.001
weight_decay= 0 # best: 0. Weight decay applies L2 regularization to the learned parameters
num_layers = 1 # best: 1f:2, 3f:1
bidirectional = True # best: True

## Extra hyperparameter for student
hidden_size_student = 64# best: 1f: 32, 3f:64
n_epochs_student = 50 # Best  3f,MLP:50
learning_rate_student = 0.0001 # best: 0.001 for SGD, 0.0001 for Adam
weight_decay_student= 0 # best 0 for Adam
student_encoder = 'MLP' # 'LSTM'

## Saving files
teacher_chkpt = '../saved_models/teacher_model_real_{}f.ckpt'.format(num_features)
teacher_plot = '../plots/teacher_predictions_real_{}f.png'.format(num_features)
teacher_plot_whole = '../plots/prediction_of_teacher_{}f.png'.format(num_features)
student_chkpt = '../saved_models/student_model_real_{}f.ckpt'.format(num_features)
student_plot = '../plots/student_predictions_real_{}f.png'.format(num_features)
student_plot_whole = '../plots/prediction_of_student_{}f.png'.format(num_features)
std_scalar_file = '../saved_models/std_scalar.pkl'
