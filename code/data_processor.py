'''
## Author: Puja Chowdhury
## Email: pujac@email.sc.edu
## Date: 07/18/2023
Processing data
'''

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import configuration as config

# Setting device
device = config.device # setting running device

def range_with_floats(start, stop, step):
    '''
    Using for sliding window
    :param start: starting point of window
    :param stop: ending point of window
    :param step: sliding length for window
    :return: yeilds the range
    '''
    while stop > start:
        yield start
        start += step

# train and test splitting
def train_test_split(t, y, split=0.8):
    '''
    Using for splitting the dataset
    :param t: time data
    :param y: features data
    :param split: split ratio
    :return: t_train, y_train, t_test, y_test
    '''

    indx_split = int(split * len(t))
    indx_train = np.arange(0, indx_split)
    indx_test = np.arange(indx_split, len(t))

    t_train = t[indx_train]
    y_train = y[:, indx_train]

    t_test = t[indx_test]
    y_test = y[:, indx_test]
    return t_train, y_train, t_test, y_test

def windowed_dataset(y, num_features=1, input_window=5, output_window=1, stride=1, std_scalar = None, return_std_scalar = False):
    '''
    Using for splitting data as windowed data
    :param y: original data
    :param num_features: features considered from y
    :param input_window: input length
    :param output_window: output length
    :param stride: slideing length
    :param std_scalar: standardization model
    :param return_std_scalar: boolean
    :return: X = input window, Y = output window, std_scalar
    '''
    y = np.transpose(y)
    if std_scalar == None:
        std_scalar = StandardScaler()
        # std_scalar =MinMaxScaler()
        y[:, 0] = std_scalar.fit_transform(y[:,0].reshape(-1, 1)).flatten()
    else:
        y[:, 0] = std_scalar.transform(y[:, 0].reshape(-1, 1)).flatten()
    for idx in range(y.shape[1]-1):
        y[:, idx+1] = std_scalar.transform(y[:, idx+1].reshape(-1, 1)).flatten()

    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1
    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, 1])

    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            try:
                X[:, ii, ff] = y[start_x:end_x, ff]
            except:
                print("Sample {} is empty for X".format(ii))
    for ii in np.arange(num_samples):
            start_y = stride * ii + input_window
            end_y = start_y + output_window
            try:
                Y[:, ii, 0] = y[start_y:end_y, 0]
            except:
                print("Sample {} is empty for Y".format(ii))

    if return_std_scalar:
        return X, Y, std_scalar
    else:
        return X, Y

def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    Converting numpy array to tensors
    :param Xtrain: numpy array
    :param Ytrain: numpy array
    :param Xtest: numpy array
    :param Ytest: numpy array
    :return: tensors
    '''

    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
