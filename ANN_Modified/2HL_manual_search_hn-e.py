import pandas as pd
import numpy as np 
import copy
from ann2 import Net
from replicate import replicate_data 
from sklearn.preprocessing import StandardScaler
from train import train

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
HN = [
    (2, 2), (2, 4), (2, 8), (2, 12), (2, 16), (2, 20), 
    (4, 2), (4, 4), (4, 8), (4, 12), (4, 16), (4, 20), 
    (8, 2), (8, 4), (8, 8), (8, 12), (8, 16), (8, 20),
    (12, 2), (12, 4), (12, 8), (12, 12), (12, 16), (12, 20),
    (16, 2), (16, 4), (16, 8), (16, 12), (16, 16), (16, 20),
    (20, 2), (20, 4), (20, 8), (20, 12), (20, 16), (20, 20)
]
EPOCHS = 3000
BATCH_SIZE = 50
LR = 0.001
MODELS = {}

for h1, h2 in HN:
    net = Net(h1, h2)
    init_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(init_state)
    training_inputs = training_data[:, 0:5]
    training_add = training_data[:, 8:]
    training_inputs = np.append(training_inputs, training_add, axis=1)
    training_labels = training_data[:, 5:8]
    
    test_inputs = testing_data[:, 0:5]
    test_add = testing_data[:, 8:]
    test_inputs = np.append(test_inputs, test_add, axis=1)
    test_labels = testing_data[:, 5:8]
    
    E_opt, opt_epochs = train(net, training_inputs, training_labels, test_inputs, test_labels, EPOCHS, LR, BATCH_SIZE)
    MODELS['{b}_{x}-{y}_{z}'.format(b=HL, x=h1, y=h2, z=opt_epochs)] = E_opt

with open('Data/Search/manual_search_results_{x}HL_hn-e_GL10.csv'.format(x=HL), 'w') as f:
    for key in MODELS.keys():
        f.write("%s: %s\n"%(key, MODELS[key]))

print(MODELS)
