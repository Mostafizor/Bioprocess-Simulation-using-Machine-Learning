import pandas as pd
import numpy as np 
import copy
from rnn import RNN
from replicate import replicate_data 
from sklearn.preprocessing import StandardScaler
from train import train
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

# Shuffle training data
np.random.shuffle(training_data)

# Manual Search Training Loop
HL = 2
HN = 10
EPOCHS = 15
BATCH_SIZE = [3, 5, 8, 10, 15, 20, 30, 40, 50, 100, 200, 300, 400, 500]
LR = 0.004
MODELS = {}

rnn = RNN(3, 5, 12, HN, HL)
init_state = copy.deepcopy(rnn.state_dict())
for bs in BATCH_SIZE:
    rnn.load_state_dict(init_state)
    training_inputs = training_data[:, 0:5]
    training_labels = training_data[:, 5:]
    test_inputs = testing_data[:, 0:5]
    test_labels = testing_data[:, 5:]

    training_inputs = np.split(training_inputs, 606)
    training_labels = np.split(training_labels, 606)
    test_inputs = np.split(test_inputs, 2)
    test_labels = np.split(test_labels, 2)

    train(rnn, training_inputs, training_labels, EPOCHS, LR, bs)
    avg_mse = test(test_inputs, test_labels, rnn)

    MODELS['{a}_{x}-{y}_{z}_{b}_{c}'.format(a=HL, x=HN, y=HN, z=EPOCHS, b=LR, c=bs)] = avg_mse

with open('Data2/Search/manual_search_results_{x}HL_bs.csv'.format(x=HL), 'w') as f:
    for key in MODELS.keys():
        f.write("%s: %s\n"%(key, MODELS[key]))

print(MODELS)
