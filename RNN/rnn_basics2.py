import torch
import torch.nn as nn
import sys

### Data Preparation ###

idx2char = ['h', 'i', 'e', 'l', 'o']

#h = [1, 0, 0, 0, 0]
#i = [0, 1, 0, 0, 0]
#e = [0, 0, 1, 0, 0]
#l = [0, 0, 0, 1, 0]
#o = [0, 0, 0, 0, 1]

# Teach hihell -> ihello
x_data = [0, 1, 0, 2, 3, 3]  # hihell
one_hot_lookup = [[1, 0, 0, 0, 0], #h
                  [0, 1, 0, 0, 0], #i
                  [0, 0, 1, 0, 0], #e
                  [0, 0, 0, 1, 0], #l   
                  [0, 0, 0, 0, 1]] #o

y_data = [1, 0, 2, 3, 3, 4]     #ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.autograd.Variable(torch.Tensor(x_one_hot))
labels = torch.autograd.Variable(torch.LongTensor(y_data))

### Parameters ###
num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5 # output size from the LSTM. 5 to directly predict one-hot
batch_size = 1  # one sentence
sequence_length = 1 # Lets do one by one
num_layers = 1 # one-layer rnn

### Model ###
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    def forward(self, x, hidden):
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, num_classes)
        return hidden, out
    
    def init_hidden(self):
        # Initialse hidden and cell states
        # (num_layers * num_directions, batch_size, hidden_size)
        return torch.autograd.Variable(torch.zeros(num_layers, batch_size, hidden_size))

### Training Loop ###
model = Model()

criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
for epoch in range(100):
    optimiser.zero_grad()
    loss = 0
    hidden = model.init_hidden()
    sys.stdout.write('Predicted String: ')
    for inputt, label in zip(inputs, labels):
        hidden, output = model(inputt, hidden)
        val, idx = output.max(1)
        sys.stdout.write(idx2char[idx.data[0]])
        loss += criterion(output, label)

    print(", epoch: %d, loss: %1.3f" % (epoch+1, loss.data[0]))

    loss.backward()
    optimiser.step()