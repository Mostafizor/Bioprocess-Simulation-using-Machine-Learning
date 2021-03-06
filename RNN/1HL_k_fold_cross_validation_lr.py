import pandas as pd
import numpy as np 
import copy
from rnn import RNN
from replicate import replicate_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from train import train
from test2 import test
import csv

# Load training data as pd dataframe and convert pd dataframe into numpy array.
training_data = pd.read_excel('Data2/reduced_training_data.xlsx')
training_data_array = np.array(training_data)

# Standardise Training Data
scaler_train = StandardScaler()
scaler_train.fit(training_data)

# Split data into k=6 folds.
kf = KFold(n_splits=6)
kf.get_n_splits(training_data)

# Split training data set into 6 subsets containing k-1 folds before optimisation.
class wrapper(object):
    def __init__(self):
        self.value = []

subset_train1 = wrapper() 
subset_train2 = wrapper()
subset_train3 = wrapper()
subset_train4 = wrapper()
subset_train5 = wrapper()
subset_train6 = wrapper()
subset_test1 = wrapper() 
subset_test2 = wrapper()
subset_test3 = wrapper()
subset_test4 = wrapper()
subset_test5 = wrapper()
subset_test6 = wrapper()
subset_train_list = [subset_train1, subset_train2, subset_train3, subset_train4, subset_train5, subset_train6]
subset_test_list = [subset_test1, subset_test2, subset_test3, subset_test4, subset_test5, subset_test6]

index = 0
for train_index, test_index in kf.split(training_data):

    for row in train_index:
        subset_train_list[index].value.append(training_data_array[row])
    
    for row in test_index:
        subset_test_list[index].value.append(training_data_array[row])
    
    index +=1

# Standardise Test Data
for subset in subset_test_list:
    subset.value = scaler_train.transform(subset.value)

# Replicate the training data in each subset.
columns = "BC NC LP LI NIC".split()
for index, subset in enumerate(subset_train_list):
    df = pd.DataFrame(data=subset.value, index=None, columns=columns)
    ref = df
    df = scaler_train.transform(df)

    replicated_data1 = replicate_data(ref, 50, 0.03)
    replicated_data1 = scaler_train.transform(replicated_data1)
    df = np.append(df, replicated_data1, axis=0) 

    replicated_data2 = replicate_data(ref, 50, 0.05)
    replicated_data2 = scaler_train.transform(replicated_data2)
    df = np.append(df, replicated_data2, axis=0) 

    subset.value = df

# Calculate training and test labels
for index1, subset in enumerate(subset_train_list):
    a = []
    
    try:
        for index2, row in enumerate(subset.value):
            dBC = subset.value[index2 + 1][0] - row[0]
            dNC = subset.value[index2 + 1][1] - row[1]
            dLP = subset.value[index2 + 1][2] - row[2]

            rates =[dBC, dNC, dLP]
            a.append(rates)
    except IndexError:
        rates = [0, 0, 0]
        a.append(rates)
    
    a = np.array(a)
    subset.value = np.append(subset.value, a, axis=1) 

for index1, subset in enumerate(subset_test_list):
    b = []
    
    try:
        for index2, row in enumerate(subset.value):
            dBC = subset.value[index2 + 1][0] - row[0] 
            dNC = subset.value[index2 + 1][1] - row[1]
            dLP = subset.value[index2 + 1][2] - row[2]

            rates =[dBC, dNC, dLP]
            b.append(rates)
    except IndexError:
        rates = [0, 0, 0]
        b.append(rates)
    
    b = np.array(b)
    subset.value = np.append(subset.value, b, axis=1)


# Remove all datapoints corresponding to 144 h from the training and testing sets
for subset in subset_train_list:
    count = 0
    decrement = 0
    for index, row in enumerate(subset.value):
        count +=1
        if count == 13:
            delete = index - decrement
            subset.value = np.delete(subset.value, delete, 0)
            decrement += 1
            count = 0

for subset in subset_test_list:
    subset.value = np.delete(subset.value, -1, 0)

subset_train_list = np.array(subset_train_list)
subset_test_list = np.array(subset_test_list)

# Shuffle Training Data
for subset in subset_train_list:
    np.random.shuffle(subset.value)

# k-fold cross validation training loop
HL = 1
HN1 = 5
EPOCHS = 30
BATCH_SIZE = 8
LR = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
MODELS = {}

rnn = RNN(3, 5, 12, HN1, HL)
init_state = copy.deepcopy(rnn.state_dict())
for lr in LR:
    MSEs = []
    for index, subset in enumerate(subset_train_list):
        subset.value = np.array(subset.value)
        subset_test_list[index].value = np.array(subset_test_list[index].value)

        rnn.load_state_dict(init_state)
        training_inputs = subset.value[:, 0:5]
        training_labels = subset.value[:, 5:]
        test_inputs = subset_test_list[index].value[:, 0:5]
        test_labels = subset_test_list[index].value[:, 5:]

        training_inputs = np.split(training_inputs, 505)
        training_labels = np.split(training_labels, 505)

        test_inputs = np.array([test_inputs])
        test_labels = np.array([test_labels])
        
        train(rnn, training_inputs, training_labels, EPOCHS, lr, BATCH_SIZE)
        avg_mse = test(test_inputs, test_labels, rnn)
        MSEs.append(avg_mse)

    avg_mse = sum(MSEs)/len(MSEs)
    MODELS['{a}_{x}_{z}_{b}'.format(a=HL, x=HN1, z=EPOCHS, b=lr)] = avg_mse

with open('Data2/Search/k_fold_results_{x}HL_lr.csv'.format(x=HL), 'w') as f:
    for key in MODELS.keys():
        f.write("%s: %s\n"%(key, MODELS[key]))

print(MODELS)