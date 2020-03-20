import torch
import numpy as np
import pandas as pd

def test(test_inputs, test_labels, net):
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

    return AVG_MSE
