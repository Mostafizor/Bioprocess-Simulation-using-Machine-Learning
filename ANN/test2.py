import torch
import numpy as np
import pandas as pd

def test(test_inputs, test_labels, net):
    net.eval()
    test_X = torch.Tensor(test_inputs).view(-1, 5)
    test_y = torch.Tensor(test_labels)

    predictionNumpy = []
    with torch.no_grad():
        for i in range(0, len(test_X)):
            net_out = net(test_X[i].view(-1, 5))
            predictionNumpy.append(net_out[0].numpy())              # The output from the net is a tensor which contains only one element which is a list. The list contains the 3 output values. We only want the list, not the tensoor containing one element which is a list.

    experimental = []
    for data in test_y:
        experimental.append(data.numpy())

    squared_error_X = []
    squared_error_N = []
    squared_error_L = [] 

    for i in range(0, len(experimental)):
            X_error = experimental[i][0] - predictionNumpy[i][0]
            N_error = experimental[i][1] - predictionNumpy[i][1]
            L_error = experimental[i][2] - predictionNumpy[i][2]
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

    predictions_online = []
    for index, value in enumerate(test_inputs):
        BC = value[0] + predictionNumpy[index][0]
        NC = value[1] + predictionNumpy[index][1]
        LP = value[2] + predictionNumpy[index][2]

        predictions_online.append([BC, NC, LP, 1, 2])

    predictions_offline = []
    BC1, BC2 = test_inputs[0][0], test_inputs[12][0]
    NC1, NC2 = test_inputs[0][1], test_inputs[12][1]
    LP1, LP2 = test_inputs[0][2], test_inputs[12][2]
    for index, value in enumerate(test_inputs):
        if index < 12:
            BC = BC1 + predictionNumpy[index][0]    ### basically fix this so that it is net(BC1, NC1...)
            NC = NC1 + predictionNumpy[index][1]
            LP = LP1 + predictionNumpy[index][2]
            predictions_offline.append([BC, NC, LP, 1, 2])
            BC1 = BC
            NC1 = NC
            LP1 = LP
        
        if index >= 12:
            BC = BC2 + predictionNumpy[index][0]
            NC = NC2 + predictionNumpy[index][1]
            LP = LP2 + predictionNumpy[index][2]
            predictions_offline.append([BC, NC, LP, 1, 2])
            BC2 = BC
            NC2 = NC
            LP2 = LP

    return AVG_MSE, predictions_online, predictions_offline
