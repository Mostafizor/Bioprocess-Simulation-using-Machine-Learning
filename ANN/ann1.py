import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(777)

class Net(nn.Module):
	'''
	This Class Defines the Structure of the Artificial Neural Network
	'''
	def __init__(self, HN):	
		self.HN = HN		
		super().__init__()                                                             # Run the intitialision method from base class nn.module.
		self.fc1 = nn.Linear(5, self.HN)                                                    # Define the first fully connected layer. nn.Linear simply connects the input nodes to the output nodes in the standard way. The input layer contains 5 nodes. The output layer (first hidden layer), consists of 15 nodes.
		self.fc2 = nn.Linear(self.HN, 3)                                                   # Hidden layer 2: each node takes in 15 values, contains 15 nodes hence outputs 15 values.

	def forward(self, x):                                                              # This method feeds data into the network and propagates it forward.
		x = torch.sigmoid(self.fc1(x))                                                 # Feed the dataset, x, through fc1 and apply the Sigmoid activation function to the weighted sum of each neuron. Then assign the transformed dataset to x. Next, feed the transformed dataset through fc2 and so on... until we reach the output layer. The activation fucntion basically decides if the neuron is 'firing' like real neurones in the human brain. The activation function prevents massive output numbers.
		x = self.fc2(x)
		return x 
