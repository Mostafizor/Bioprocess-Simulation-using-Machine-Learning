import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable

#### NOTE: you know how net_out, test_inputs and test_labels look, Now redesign test function

def test(test_inputs, test_labels, net):
    net.eval()
    test_X = Variable(torch.Tensor(test_inputs)) 
    print(test_X)
    test_y = Variable(torch.Tensor(test_labels))
    print(test_y)

    predictionNumpy = []
    with torch.no_grad():

        net_out = net(test_X)
        print(net_out)
