'''
## Author: Puja Chowdhury
## Email: pujac@email.sc.edu
## Date: 07/18/2023
Functions for assessment
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import PIMENTO
from sklearn.metrics import mean_squared_error, mean_absolute_error
import configuration as config
# Setting device
device = config.device

def mse_calculation(model, Xtest, Ytest, target_len, student_mode = False, rmse = False, std_scalar=None):
    mse_all = []
    for i in range(Xtest.shape[1]):
        X_test_plt = Xtest[:, i, :]
        Y_test_plt = Ytest[:, i, :]
        if student_mode:
            Y_test_pred = PIMENTO.predict(model,
                                          torch.from_numpy(X_test_plt).type(torch.Tensor)[:, 0].unsqueeze(-1))
        else:
            Y_test_pred = PIMENTO.predict(model, torch.from_numpy(X_test_plt).type(torch.Tensor))
        if std_scalar == None:
            mse = mean_squared_error(Y_test_plt, Y_test_pred, squared=rmse)
        else:
            Y_truth_test = std_scalar.inverse_transform(Y_test_plt.reshape(-1, 1)).flatten()
            Y_Pred_test = std_scalar.inverse_transform(Y_test_pred.reshape(-1, 1)).flatten()
            mse = mean_squared_error(Y_truth_test, Y_Pred_test, squared=rmse)
        mse_all.append(mse)
    return sum(mse_all)/len(mse_all)

def whole_series_prediction(model,X_IN,Y_Truth, std_scalar, target_len, student_mode = False, rmse = False):
    for i in range(X_IN.shape[1]):
        X_test_plt = X_IN[:, i, :]
        Y_test_plt = Y_Truth[:, i, :]
        if student_mode:
            Y_test_pred = PIMENTO.predict(model,
                                          torch.from_numpy(X_test_plt).type(torch.Tensor)[:, 0].unsqueeze(-1))
        else:
            Y_test_pred = PIMENTO.predict(model, torch.from_numpy(X_test_plt).type(torch.Tensor))
        if i==0:
            Y_Truth_All = Y_test_plt
            Y_Pred_All = Y_test_pred

        else:
            Y_Truth_All = np.concatenate((Y_Truth_All,Y_test_plt), axis = 0)
            Y_Pred_All = np.concatenate((Y_Pred_All,Y_test_pred), axis = 0)
    Y_Truth_All = std_scalar.inverse_transform(Y_Truth_All.reshape(-1, 1)).flatten()
    Y_Pred_All = std_scalar.inverse_transform(Y_Pred_All.reshape(-1, 1)).flatten()
    mse_error = mean_squared_error(Y_Truth_All, Y_Pred_All, squared=rmse)
    mae_error = mean_absolute_error(Y_Truth_All, Y_Pred_All)
    return Y_Truth_All, Y_Pred_All, mse_error, mae_error

def plot_train_test_results(lstm_model, Xtrain, Ytrain, Xtest, Ytest, num_rows=4, filename='../plots/predictions.png', student_version=False):

    # input window size
    iw = Xtrain.shape[0]
    ow = Ytrain.shape[0]

    # figure setup
    num_cols = 2
    num_plots = num_rows * num_cols

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(13, 15))

    # plot training/test predictions
    for ii in range(num_rows):
        # train set
        X_train_plt = Xtrain[:, ii, :]
        if student_version:
            Y_train_pred = PIMENTO.predict(lstm_model,
                                           torch.from_numpy(X_train_plt).type(torch.Tensor)[:, 0].unsqueeze(-1))
        else:
            Y_train_pred = PIMENTO.predict(lstm_model, torch.from_numpy(X_train_plt).type(torch.Tensor).to(device))

        ax[ii, 0].plot(np.arange(0, iw), Xtrain[:, ii, 0], 'k', linewidth=2, label='Input')
        ax[ii, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, ii, 0]], Ytrain[:, ii, 0]]),
                       color=(0.2, 0.42, 0.72), linewidth=2, label='Target')
        ax[ii, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, ii, 0]], Y_train_pred[:, 0]]),
                       color=(0.76, 0.01, 0.01), linewidth=2, label='Prediction')
        # ax[ii, 0].plot(Ytrain[:, ii, 0], color=(0.2, 0.42, 0.72), linewidth=2, label='Target')
        # ax[ii, 0].plot(Y_train_pred[:, 0], color=(0.76, 0.01, 0.01), linewidth=2, label='Prediction')
        ax[ii, 0].set_xlim([0, iw + ow - 1])
        ax[ii, 0].set_ylim([-2, 2])
        ax[ii, 0].set_xlabel('$t$')
        ax[ii, 0].set_ylabel('$y$')

        # test set
        X_test_plt = Xtest[:, ii, :]
        if student_version:
            Y_test_pred = PIMENTO.predict(lstm_model,
                                          torch.from_numpy(X_test_plt).type(torch.Tensor)[:, 0].unsqueeze(-1))
        else:
            Y_test_pred = PIMENTO.predict(lstm_model, torch.from_numpy(X_test_plt).type(torch.Tensor))

        ax[ii, 1].plot(np.arange(0, iw), Xtest[:, ii, 0], 'k', linewidth=2, label='Input')
        ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Ytest[:, ii, 0]]),
                       color=(0.2, 0.42, 0.72), linewidth=2, label='Target')
        ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Y_test_pred[:, 0]]),
                       color=(0.76, 0.01, 0.01), linewidth=2, label='Prediction')
        # ax[ii, 1].plot(Ytest[:, ii, 0], color=(0.2, 0.42, 0.72), linewidth=2, label='Target')
        # ax[ii, 1].plot(Y_test_pred[:, 0], color=(0.76, 0.01, 0.01), linewidth=2, label='Prediction')
        ax[ii, 1].set_xlim([0, iw + ow - 1])
        ax[ii, 1].set_ylim([-2, 2])
        ax[ii, 1].set_xlabel('$t$')
        ax[ii, 1].set_ylabel('$y$')

        if ii == 0:
            ax[ii, 0].set_title('Train')

            ax[ii, 1].legend(bbox_to_anchor=(1, 1))
            ax[ii, 1].set_title('Test')

    plt.suptitle('LSTM Encoder-Decoder Predictions', x=0.445, y=1.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(filename)
    plt.show()
    return
