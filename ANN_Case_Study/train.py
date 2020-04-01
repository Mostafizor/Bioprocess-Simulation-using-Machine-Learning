import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from test import test

def train(net, training_inputs, training_labels, test_inputs, test_labels, EPOCHS, l_rate, BATCH_SIZE):
	net.train()                                                                         
	optimiser = optim.Adam(net.parameters(), lr = l_rate)									   # net.parameters(): all of the adjustable parameters in our network. lr: a hyperparameter adjusts the size of the step that the optimizer will take to minimise the loss.
	loss_function = nn.MSELoss(reduction='mean')

	X = torch.Tensor(training_inputs).view(-1, 4)
	y = torch.Tensor(training_labels)

	E_va_list = []
	GL_MAX = 3
	for epoch in range(EPOCHS):
		for i in tqdm(range(0, len(X), BATCH_SIZE)):
			batch_X = X[i:i+BATCH_SIZE].view(-1, 4)
			batch_y = y[i:i+BATCH_SIZE]

			optimiser.zero_grad()
			outputs = net(batch_X)
			loss = loss_function(outputs, batch_y)
			loss.backward()
			optimiser.step()

		E_va = test(test_inputs, test_labels, net)
		E_va_list.append(E_va)

		GL = 100*((E_va/min(E_va_list)) - 1)

		if GL > GL_MAX:
			return min(E_va_list), (E_va_list.index(min(E_va_list)) + 1)
	
	return min(E_va_list), (E_va_list.index(min(E_va_list)) + 1)
