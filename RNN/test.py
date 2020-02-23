import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable

def test(test_inputs, test_labels, net, BATCH_SIZE):
    net.eval()
    test_X = Variable(torch.Tensor(test_inputs)) 
    test_y = Variable(torch.Tensor(test_labels))

    hidden = net.init_hidden(test_X)
    with torch.no_grad():
        net_out, _ = net(test_X, hidden)

    squared_error_X = []
    squared_error_N = []
    squared_error_L = [] 

    for index1, element in enumerate(test_y):
        for index2, row in enumerate(element):
            X_error = row[0] - net_out[index1][index2][0]
            N_error = row[1] - net_out[index1][index2][1]
            L_error = row[2] - net_out[index1][index2][2]
            squared_error_X.append(X_error**2)
            squared_error_N.append(N_error**2)
            squared_error_L.append(L_error**2)

    MSE_X1 = sum(squared_error_X[0:12])/12
    MSE_N1 = sum(squared_error_N[0:12])/12
    MSE_L1 = sum(squared_error_L[0:12])/12
    MSE_X2 = sum(squared_error_X[12:24])/12
    MSE_N2 = sum(squared_error_N[12:24])/12
    MSE_L2 = sum(squared_error_L[12:24])/12
    MSE_list = [MSE_X1, MSE_N1, MSE_L1, MSE_X2, MSE_N2, MSE_L2]
    AVG_MSE = sum(MSE_list)/6

    LI1, LI2 = test_X[0][0][3], test_X[1][0][3]
    NIC1, NIC2 = test_X[0][0][4], test_X[1][0][4]
    predictions_online = []
    for index1, element in enumerate(test_X):
        if index1 == 0:
            for index2, row in enumerate(element):
                BC = row[0] + net_out[index1][index2][0]
                NC = row[1] + net_out[index1][index2][1]
                LP = row[2] + net_out[index1][index2][2]

                predictions_online.append([BC, NC, LP, LI1, NIC1])
        
        if index1 == 1:
            for index2, row in enumerate(element):
                BC = row[0] + net_out[index1][index2][0]
                NC = row[1] + net_out[index1][index2][1]
                LP = row[2] + net_out[index1][index2][2]

                predictions_online.append([BC, NC, LP, LI2, NIC2])
    predictions_online = np.array(predictions_online)

    predictions_offline = []
    BC1, BC2 = test_X[0][0][0], test_X[1][0][0]
    NC1, NC2 = test_X[0][0][1], test_X[1][0][1]
    LP1, LP2 = test_X[0][0][2], test_X[1][0][2]
    net.sequence_length = 1                         # We will now be feeding 1 input at a time to the network(offline prediction), so i will need to start extracting and feeding hidden state per item in a sequence. 
    for index1, element in enumerate(test_X):
        hidden = net.init_hidden(Variable(torch.Tensor([[[]]])))            # Initialise hidden state with a batch size of 1

        for index2, row in enumerate(element):

            if index1 == 0:
                # Feed inputs with a batch size of 1, sequence length of 1 and feature vector length of 5 to the network
                net_out, hidden = net(Variable(torch.Tensor([    
                    [[BC1, NC1, LP1, LI1, NIC1]]
                ])), hidden)
                BC = BC1 + net_out[0][0][0]
                NC = NC1 + net_out[0][0][1]
                LP = LP1 + net_out[0][0][2]
                predictions_offline.append([float(BC), float(NC), float(LP), float(LI1), float(NIC1)])
                BC1 = BC
                NC1 = NC
                LP1 = LP
            
            if index1 == 1:
                net_out, hidden = net(Variable(torch.Tensor([    
                    [[BC2, NC2, LP2, LI2, NIC2]]
                ])), hidden)
                BC = BC2 + net_out[0][0][0]
                NC = NC2 + net_out[0][0][1]
                LP = LP2 + net_out[0][0][2]
                predictions_offline.append([float(BC), float(NC), float(LP), float(LI2), float(NIC2)])
                BC2 = BC
                NC2 = NC
                LP2 = LP
    predictions_offline = np.array(predictions_offline)

    return AVG_MSE, predictions_online, predictions_offline
