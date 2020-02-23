import torch
import pandas as pd
import numpy as np 
from rnn import RNN
from replicate import replicate_data 
from sklearn.preprocessing import StandardScaler
from train import train
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

HL = 1
HN1 = 4
EPOCHS = 15
LR = 0.001
BATCH_SIZE = 5
rnn = RNN(3, 5, 12, HN1, HL)

avg_mse=1
while avg_mse > 0.9:
    training_inputs = training_data[:, 0:5]
    training_labels = training_data[:, 5:]
    test_inputs = testing_data[:, 0:5]
    test_labels = testing_data[:, 5:]

    training_inputs = np.split(training_inputs, 606)
    training_labels = np.split(training_labels, 606)
    test_inputs = np.split(test_inputs, 2)
    test_labels = np.split(test_labels, 2)


    train(rnn, training_inputs, training_labels, EPOCHS, LR, BATCH_SIZE)
    avg_mse, predictions_online, predictions_offline = test(test_inputs, test_labels, rnn, BATCH_SIZE)
print(avg_mse)

predictions_online_inverse_transform = scaler_test.inverse_transform(predictions_online)
predictions_offline_inverse_transform = scaler_test.inverse_transform(predictions_offline)

online = pd.DataFrame(predictions_online_inverse_transform)
offline = pd.DataFrame(predictions_offline_inverse_transform)
avg_mse = pd.DataFrame([avg_mse, 0])


online.to_excel('Data/Optimised_Networks/online {x}_{y}_{a}_{b}_{c}.xlsx'.format(x=HL, y=HN1, a=EPOCHS, b=LR, c=BATCH_SIZE))
offline.to_excel('Data/Optimised_Networks/offline {x}_{y}_{a}_{b}_{c}.xlsx'.format(x=HL, y=HN1, a=EPOCHS, b=LR, c=BATCH_SIZE))
avg_mse.to_excel('Data/Optimised_Networks/avg_mse {x}_{y}_{a}_{b}_{c}.xlsx'.format(x=HL, y=HN1, a=EPOCHS, b=LR, c=BATCH_SIZE))

torch.save(rnn.state_dict(), 'Data/Optimised_Networks/Models/TTT {x}_{y}_{a}_{b}_{c}.pt'.format(x=HL, y=HN1, a=EPOCHS, b=LR, c=BATCH_SIZE))