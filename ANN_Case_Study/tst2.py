import torch
import numpy as np
import pandas as pd

def test(test_inputs, test_labels, net):
    net.eval()
    test_X = torch.Tensor(test_inputs).view(-1, 4)
    test_y = torch.Tensor(test_labels)

    predictionNumpy = []
    with torch.no_grad():
        for i in range(0, len(test_X)):
            net_out = net(test_X[i].view(-1, 4))
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

    MSE_X1 = sum(squared_error_X[0:25])/25
    MSE_N1 = sum(squared_error_N[0:25])/25
    MSE_L1 = sum(squared_error_L[0:25])/25
    MSE_X2 = sum(squared_error_X[25:50])/25
    MSE_N2 = sum(squared_error_N[25:50])/25
    MSE_L2 = sum(squared_error_L[25:50])/25
    MSE_X3 = sum(squared_error_X[50:75])/25
    MSE_N3 = sum(squared_error_N[50:75])/25
    MSE_L3 = sum(squared_error_L[50:75])/25
    MSE_X4 = sum(squared_error_X[75:100])/25
    MSE_N4 = sum(squared_error_N[75:100])/25
    MSE_L4 = sum(squared_error_L[75:100])/25
    MSE_list = [MSE_X1, MSE_N1, MSE_L1, MSE_X2, MSE_N2, MSE_L2, MSE_X3, MSE_N3, MSE_L3, MSE_X4, MSE_N4, MSE_L4]
    AVG_MSE = sum(MSE_list)/12

    LI1, LI2, LI3, LI4 = test_inputs[0][3], test_inputs[25][3], test_inputs[50][3], test_inputs[75][3]
    predictions_online = []
    for index, value in enumerate(test_inputs):
        BC = value[0] + predictionNumpy[index][0]
        NC = value[1] + predictionNumpy[index][1]
        LP = value[2] + predictionNumpy[index][2]

        if index < 25:
            predictions_online.append([BC, NC, LP, LI1])

        if index in range(25, 50):
            predictions_online.append([BC, NC, LP, LI2])
        
        if index in range(50, 75):
            predictions_online.append([BC, NC, LP, LI3])
        
        if index in range(75, 100):
            predictions_online.append([BC, NC, LP, LI4])


    predictions_offline = []
    BC1, BC2, BC3, BC4 = test_inputs[0][0], test_inputs[25][0], test_inputs[50][0], test_inputs[75][0]
    NC1, NC2, NC3, NC4 = test_inputs[0][1], test_inputs[25][1], test_inputs[50][1], test_inputs[75][1]
    LP1, LP2, LP3, LP4 = test_inputs[0][2], test_inputs[25][2], test_inputs[50][2], test_inputs[75][2]
    for index, value in enumerate(test_inputs):
        if index < 25:
            net_out = net(torch.Tensor([BC1, NC1, LP1, LI1]))
            BC = BC1 + net_out[0]   
            NC = NC1 + net_out[1]
            LP = LP1 + net_out[2]
            predictions_offline.append([float(BC), float(NC), float(LP), float(LI1)])
            BC1 = BC
            NC1 = NC
            LP1 = LP
        
        if index in range(25, 50):
            net_out = net(torch.Tensor([BC2, NC2, LP2, LI2]))
            BC = BC2 + net_out[0] 
            NC = NC2 + net_out[1] 
            LP = LP2 + net_out[2] 
            predictions_offline.append([float(BC), float(NC), float(LP), float(LI2)])
            BC2 = BC
            NC2 = NC
            LP2 = LP
        
        if index in range(50, 75):
            net_out = net(torch.Tensor([BC3, NC3, LP3, LI3]))
            BC = BC3 + net_out[0] 
            NC = NC3 + net_out[1] 
            LP = LP3 + net_out[2] 
            predictions_offline.append([float(BC), float(NC), float(LP), float(LI3)])
            BC3 = BC
            NC3 = NC
            LP3 = LP
        
        if index in range(75, 100):
            net_out = net(torch.Tensor([BC4, NC4, LP4, LI4]))
            BC = BC4 + net_out[0] 
            NC = NC4 + net_out[1] 
            LP = LP4 + net_out[2] 
            predictions_offline.append([float(BC), float(NC), float(LP), float(LI4)])
            BC4 = BC
            NC4 = NC
            LP4 = LP
            
    return AVG_MSE, predictions_online, predictions_offline
