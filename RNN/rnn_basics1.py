import torch
import torch.nn as nn

# One cell RNN input_dim (4) -> output_dim (2)
cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

h = [1, 0, 0, 0, 0]
i = [0, 1, 0, 0, 0]
e = [0, 0, 1, 0, 0]
l = [0, 0, 0, 1, 0]
o = [0, 0, 0, 0, 1]

# One letter input
inputs = torch.autograd.Variable(torch.Tensor([[h, e, l, l, o],
                                               [e, o, l, l, l],
                                               [l, l, e, e, l]]))  # shape: (3, 5, 4) ; shape for my data would be (3, 12, 5) where 3 signifies the batch size. If i wanted a batch size of 50, the shape would be (50, 12, 5)

# Initialise the hidden state wit random values
# (num_layers * num_directions, batch_size, hidden_size)
hidden = torch.autograd.Variable(torch.randn(1, 3, 2))

# Feed one element at a time
# After each step, hidden contains the hidden state
out, hidden = cell(inputs, hidden)
print('out', out.data)