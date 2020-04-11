import pandas as pd
import numpy as np 
import copy
from rnn import RNN
from replicate import replicate_data 
from sklearn.preprocessing import StandardScaler
from train2 import train
from test2 import test

# Load training and testing data as pd dataframe
training_data = pd.read_excel('Data2/reduced_training_data.xlsx')
testing_data = pd.read_excel('Data2/test_data.xlsx')

# Standardise training and testing data
scaler_train = StandardScaler()
scaler_test = StandardScaler()

scaler_train.fit(training_data)
scaler_test.fit(testing_data)

testing_data = scaler_test.transform(testing_data)

# Convert training data to pd dataframe
columns = "BC NC LP LI NIC".split()
training_data = pd.DataFrame(data=training_data, index=None, columns=columns)

# Replicate the training data
replicated_data1 = replicate_data(training_data, 50, 0.03)
replicated_data2 = replicate_data(training_data, 50, 0.05)

training_data = training_data.append(replicated_data1, ignore_index=True, sort=False)
training_data = training_data.append(replicated_data2, ignore_index=True, sort=False)

training_data = scaler_train.transform(training_data)
training_data = np.array(training_data)

# Calculate training and testing labels
try:
    a = []
    for index, row in enumerate(training_data):
        dBC = training_data[index + 1][0] - row[0]
        dNC = training_data[index + 1][1] - row[1]
        dLP = training_data[index + 1][2] - row[2]
        
        rates = [dBC, dNC, dLP]
        a.append(rates)
except IndexError:
    rates = [0, 0, 0]
    a.append(rates)

a = np.array(a)
training_data = np.append(training_data, a, axis=1)

try:
    a = []
    for index, row in enumerate(testing_data):
        dBC = testing_data[index + 1][0] - row[0]
        dNC = testing_data[index + 1][1] - row[1]
        dLP = testing_data[index + 1][2] - row[2]
        
        rates = [dBC, dNC, dLP]
        a.append(rates)
except IndexError:
    rates = [0, 0, 0]
    a.append(rates)

a = np.array(a)
testing_data = np.append(testing_data, a, axis=1)

# Remove all datapoints corresponding to 144 h from the training and testing sets
count = 0
decrement = 0
for index, row in enumerate(training_data):
    count += 1
    if count == 13:
        delete = index - decrement
        training_data = np.delete(training_data, delete, 0)
        decrement += 1
        count = 0

count = 0
decrement = 0
for index, row in enumerate(testing_data):
    count += 1
    if count == 13:
        delete = index - decrement
        testing_data = np.delete(testing_data, delete, 0)
        decrement += 1
        count = 0


# Manual Search Training Loop
HL = 1
HN1 = 15
EPOCHS = 30
BATCH_SIZE = 8
LR = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
MODELS = {}

rnn = RNN(3, 5, 12, HN1, HL)
init_state = copy.deepcopy(rnn.state_dict())
for lr in LR:
    rnn.load_state_dict(init_state)
    training_inputs = training_data[:, 0:5]
    training_labels = training_data[:, 5:]
    test_inputs = testing_data[:, 0:5]
    test_labels = testing_data[:, 5:]

    training_inputs = np.split(training_inputs, 606)
    training_labels = np.split(training_labels, 606)
    test_inputs = np.split(test_inputs, 2)
    test_labels = np.split(test_labels, 2)

    train(rnn, training_inputs, training_labels, EPOCHS, lr, BATCH_SIZE)
    avg_mse = test(test_inputs, test_labels, rnn)

    MODELS['{a}_{x}_{z}_{b}'.format(a=HL, x=HN1, z=EPOCHS, b=lr)] = np.array(avg_mse)

with open('Data2/Search/manual_search_results_{x}HL_lr_GLMAX1_1_15_30.csv'.format(x=HL), 'w') as f:
    for key in MODELS.keys():
        f.write("%s: %s\n"%(key, MODELS[key]))

print(MODELS)
