import torch
import torch.nn as nn
from torch.autograd import Variable

y_data = [1, 0, 2, 3, 3, 4] 
labels = Variable(torch.LongTensor(y_data))

print(labels)