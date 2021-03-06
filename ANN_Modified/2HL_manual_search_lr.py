import pandas as pd
import numpy as np 
import copy
from ann2 import Net
from replicate import replicate_data 
from sklearn.preprocessing import StandardScaler
from train2 import train
from test import test

# Load training and testing data as pd dataframe
training_data = pd.read_excel('Data/reduced_training_data.xlsx')
testing_data = pd.read_excel('Data/test_data.xlsx')

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

# Calculate Input Rates of Change
a = []
counter = 0
for index, row in enumerate(training_data):

    if counter == 0:
        i_rates = [0, 0, 0]
        a.append(i_rates)
        counter+=1
    elif counter == 12:
        dBC = training_data[index-1][5]
        dNC = training_data[index-1][6]
        dLP = training_data[index-1][7]
    
        i_rates = [dBC, dNC, dLP]
        a.append(i_rates)
        counter = 0
    else: 
        dBC = training_data[index-1][5]
        dNC = training_data[index-1][6]
        dLP = training_data[index-1][7]
    
        i_rates = [dBC, dNC, dLP]
        a.append(i_rates)
        counter+=1

a = np.array(a)
training_data = np.append(training_data, a, axis=1)

a = []
counter = 0
for index, row in enumerate(testing_data):

    if counter == 0:
        i_rates = [0, 0, 0]
        a.append(i_rates)
        counter+=1
    elif counter == 12:
        dBC = testing_data[index-1][5]
        dNC = testing_data[index-1][6]
        dLP = testing_data[index-1][7]
    
        i_rates = [dBC, dNC, dLP]
        a.append(i_rates)
        counter = 0
    else: 
        dBC = testing_data[index-1][5]
        dNC = testing_data[index-1][6]
        dLP = testing_data[index-1][7]
    
        i_rates = [dBC, dNC, dLP]
        a.append(i_rates)
        counter+=1

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
HN1, HN2 = 4, 4
EPOCHS = 106
BATCH_SIZE = 50
LR = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
MODELS = {}

net = Net(HN1, HN2)
init_state = copy.deepcopy(net.state_dict())
for lr in LR:
    net.load_state_dict(init_state)
    training_inputs = training_data[:, 0:5]
    training_add = training_data[:, 8:]
    training_inputs = np.append(training_inputs, training_add, axis=1)
    training_labels = training_data[:, 5:8]
    
    test_inputs = testing_data[:, 0:5]
    test_add = testing_data[:, 8:]
    test_inputs = np.append(test_inputs, test_add, axis=1)
    test_labels = testing_data[:, 5:8]
    
    train(net, training_inputs, training_labels, EPOCHS, lr, BATCH_SIZE)
    avg_mse = test(test_inputs, test_labels, net)

    MODELS['{a}_{x}-{y}_{z}_{b}'.format(a=HL, x=HN1, y=HN2, z=EPOCHS, b=lr)] = avg_mse

with open('Data/Search/manual_search_results_{x}HL_lr.csv'.format(x=HL), 'w') as f:
    for key in MODELS.keys():
        f.write("%s: %s\n"%(key, MODELS[key]))

print(MODELS)
