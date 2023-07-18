'''
## Author: Puja Chowdhury
## Email: pujac@email.sc.edu
## Date: 07/18/2023
PIMENTO: Physics-Informed Machine LEarning non-stationary Temporal Forecasting
'''

import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import copy
import configuration as config

# Setting device
device = config.device


def train_model(model, input_tensor, target_tensor, n_epochs, batch_size,learning_rate=0.01, weight_decay=0.00001):
    # initialize array of losses
    losses = np.full(n_epochs, np.nan)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)  # weight_decay=config.weight_decay
    # criterion = nn.MSELoss() # l2 loss
    criterion = nn.L1Loss() # MAE loss # better than mse

    # calculate number of batch iterations
    n_batches = int(input_tensor.shape[1] / batch_size)

    with trange(n_epochs) as tr:
        for it in tr:
            batch_loss = 0.
            for b in range(n_batches):
                # select data
                input_batch = input_tensor[:, b: b + batch_size, :]
                target_batch = target_tensor[:, b: b + batch_size, :]

                # zero the gradient
                optimizer.zero_grad()
                outputs = model(input_batch)

                # compute the loss
                loss = criterion(outputs, target_batch)
                batch_loss += loss.item()

                # backpropagation
                loss.backward()
                optimizer.step()

            # loss for epoch
            batch_loss /= n_batches
            losses[it] = batch_loss

            # progress bar
            tr.set_postfix(loss="{0:.3f}".format(batch_loss))

    return losses


def train_student_model(teacher_model, student_model, input_tensor, target_tensor, n_epochs, batch_size, learning_rate = 0.01, weight_decay = 0.00001):

    student_model.decoder.load_state_dict(copy.deepcopy(teacher_model.decoder.state_dict()))
    for param in student_model.decoder.parameters():
        param.requires_grad = False # this portion of the model will not train during this process
    for param in teacher_model.parameters():
        param.requires_grad = False  # this portion of the model will not train during this process
    # initialize array of losses
    losses = np.full(n_epochs, np.nan)

    optimizer = optim.Adam(student_model.encoder.parameters(), lr=learning_rate, weight_decay = weight_decay)  # weight_decay=config.weight_decay
    # optimizer = optim.AdamW(student_model.encoder.parameters(), lr=learning_rate)  # weight_decay=config.weight_decay

    criterion_mse = nn.MSELoss() # MSE loss
    criterion_mae = nn.L1Loss() # MAE loss

    # calculate number of batch iterations
    n_batches = int(input_tensor.shape[1] / batch_size)

    with trange(n_epochs) as tr:
        for it in tr:
            batch_loss = 0.
            for b in range(n_batches):
                # select data
                input_batch = input_tensor[:, b: b + batch_size, :]
                target_batch = target_tensor[:, b: b + batch_size, :]

                # zero the gradient
                optimizer.zero_grad()

                # encoder outputs
                encoder_output_teacher, encoder_hidden_teacher = teacher_model.encoder(input_batch)
                if config.student_encoder == 'MLP':
                    encoder_output_student = student_model.encoder(input_batch[:, :, 0].unsqueeze(-1)) # for MLP
                else:
                    encoder_output_student = student_model.encoder(input_batch[:, :, 0].unsqueeze(-1))[0] # for LSTM

                outputs = student_model(input_batch[:, :, 0].unsqueeze(-1))

                # compute the loss
                loss_encoder = criterion_mse(encoder_output_teacher, encoder_output_student)
                loss_model = criterion_mae(outputs, target_batch)
                loss = loss_encoder+loss_model
                batch_loss += loss.item()

                # backpropagation
                loss.backward()
                optimizer.step()

            # loss for epoch
            batch_loss /= n_batches
            losses[it] = batch_loss

            # progress bar
            tr.set_postfix(loss="{0:.3f}".format(batch_loss))

    return losses


def predict(model, input_tensor):
    # encode input_tensor
    input_tensor = input_tensor.unsqueeze(1) .to(device)    # add in batch size of 1
    if model.student_mode:
        if config.student_encoder == 'MLP':
            encoder_output = model.encoder(input_tensor) # MLP
        else:
            encoder_output = model.encoder(input_tensor)[0]  # LSTM

        # decoder outputs
        outputs, decoder_hidden = model.decoder(encoder_output)
    else:
        encoder_output, encoder_hidden = model.encoder(input_tensor)
        # decoder outputs
        outputs, decoder_hidden = model.decoder(encoder_output, encoder_hidden)

    np_outputs = outputs.squeeze(-1).cpu().detach().numpy()

    return np_outputs

class lstm_encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, encoder_dim, layer_dim=config.num_layers, bidirectional=True):
        super(lstm_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.encoder_dim = encoder_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1

        # RNN
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=bidirectional)
        # tensor containing the output features (h_t) from the last layer of the LSTM
        # Readout layer
        self.fc = nn.Linear(hidden_dim * self.direction, encoder_dim)

    def forward(self, x_input):
        # Forward propagate RNN
        rnn_out, self.hidden = self.rnn(x_input)
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out = F.selu(self.fc(rnn_out))
        return out, self.hidden

    def init_hidden(self, x_input):
        '''
        This function init_hidden() doesn’t initialize weights,
        it creates new initial states for new sequences.
        There’s initial state in all RNNs to calculate hidden state at time t=1.
        You can check size of this hidden variable to confirm this.
        '''
        # Set initial states
        # Not necessary. Only use if you want to use random initialization
        h0 = torch.randn(self.layer_dim * self.direction, x_input.size(0), self.hidden_dim).to(device)  # 2 for bidirection
        c0 = torch.randn(self.layer_dim * self.direction, x_input.size(0), self.hidden_dim).to(device)
        self.hidden = (h0, c0)
        return self.hidden

class lstm_decoder(nn.Module):

    def __init__(self, time_input, time_output, input_dim, hidden_dim, decoder_dim=1, layer_dim=config.num_layers,
                 bidirectional=True):
        super(lstm_decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.decoder_dim = decoder_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1

        # RNN
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=bidirectional)
        # tensor containing the output features (h_t) from the last layer of the LSTM
        # Readout layer
        self.fc_0 = nn.Linear(time_input, time_output)
        self.fc_1 = nn.Linear(hidden_dim * self.direction, decoder_dim)

    def forward(self, x_input, encoder_hidden_states=None):
        # Forward propagate RNN
        if encoder_hidden_states==None:
            rnn_out, self.hidden = self.rnn(x_input)
        else:
            rnn_out, self.hidden = self.rnn(x_input, encoder_hidden_states)
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # input time length to output time length conversion
        length_conversion_out = self.fc_0(rnn_out.clone().transpose_(0, 2)).clone().transpose_(0, 2)
        out = F.selu(self.fc_1(length_conversion_out))
        return out, self.hidden

class lstm_seq2seq(nn.Module):

    def __init__(self, input_length, output_length, input_size, hidden_size, encoder_size, layer_dim, bidirectional=False, student_mode = False):
        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_size = encoder_size
        self.student_mode = student_mode

        if self.student_mode:
            if config.student_encoder == 'MLP':
                self.encoder = mlp_encoder(input_dim=input_size, hidden_dim=config.hidden_size_student, encoder_dim=encoder_size)
            else:
                self.encoder = lstm_encoder(input_dim=input_size, hidden_dim=hidden_size, encoder_dim=encoder_size,
                                       layer_dim=layer_dim, bidirectional=bidirectional)
        else:
            self.encoder = lstm_encoder(input_dim=input_size, hidden_dim=hidden_size, encoder_dim = encoder_size,
                            layer_dim=layer_dim, bidirectional=bidirectional )
        self.decoder = lstm_decoder(input_dim=encoder_size, hidden_dim=hidden_size, decoder_dim = 1, time_input = input_length,
                                    time_output = output_length, layer_dim=layer_dim, bidirectional=bidirectional )

    def forward(self, input_batch):
        # encoder outputs
        if self.student_mode:
            if config.student_encoder == 'MLP':
                encoder_output = self.encoder(input_batch)
            else:
                encoder_output = self.encoder(input_batch)[0]
            # decoder outputs
            decoder_output, decoder_hidden = self.decoder(encoder_output)
        else:
            encoder_output, encoder_hidden = self.encoder(input_batch)
            # decoder outputs
            decoder_output, decoder_hidden = self.decoder(encoder_output, encoder_hidden)

        return decoder_output

class mlp_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder_dim):
        super(mlp_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        # Readout layer
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        # self.fc_1_1 = nn.Linear(hidden_dim, 16)
        self.fc_2 = nn.Linear(hidden_dim, encoder_dim)
        # self.actL = torch.nn.LeakyReLU(0.1)

    def forward(self, x_input):
        # Forward propagate
        # out = F.silu(self.fc_2(self.fc_1(x_input)))
        out = F.elu(self.fc_2(self.fc_1(x_input)))
        # out = F.elu(self.fc_2(self.fc_1_1(self.fc_1(x_input))))
        # out = self.actL(self.fc_2(self.fc_1(x_input)))
        return out