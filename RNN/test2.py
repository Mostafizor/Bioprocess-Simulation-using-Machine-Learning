import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable

#### NOTE: you know how net_out, test_inputs and test_labels look, Now redesign test function

def test(test_inputs, test_labels, net):
    net.eval()
    test_X = Variable(torch.Tensor(test_inputs)) 
    test_y = Variable(torch.Tensor(test_labels))

    predictionNumpy = []
    with torch.no_grad():

        net_out = net(test_X)

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

    return AVG_MSE
