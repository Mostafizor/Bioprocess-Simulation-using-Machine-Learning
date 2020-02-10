import pandas as pd
import numpy as np 
from ann2 import Net
from replicate import replicate_data
from sklearn import preprocessing 
from sklearn.model_selection import KFold
from train import train
from test import test
import csv

# Load training data as pd dataframe and convert pd dataframe into numpy array.
training_data = pd.read_excel('Data/reduced_training_data.xlsx')
training_data_array = np.array(training_data)

# Standardise Training Data
training_data_array = preprocessing.scale(training_data_array)

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

# Replicate the training data in each subset.
columns = "BC NC LP LI NIC".split()
for index, subset in enumerate(subset_train_list):
    df = pd.DataFrame(data=subset.value, index=None, columns=columns)
    ref = df

    replicated_data1 = replicate_data(ref, 50, 0.03)
    df = df.append(replicated_data1, ignore_index=True, sort=False)

    replicated_data2 = replicate_data(ref, 50, 0.05)
    df = df.append(replicated_data2, ignore_index=True, sort=False)

    subset.value = np.array(df)

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
HL = 2
HN1 = 15
HN2 = 9
EPOCHS = 30
BATCH_SIZE = [5, 10, 15, 20, 30, 40, 50, 100, 200, 300, 400, 500]
LR = 0.001
MODELS = {}

for bs in BATCH_SIZE:
    MSEs = []
    for index, subset in enumerate(subset_train_list):
        subset.value = np.array(subset.value)
        subset_test_list[index].value = np.array(subset_test_list[index].value)

        net = Net(HN1, HN2)
        training_inputs = subset.value[:, 0:5]
        training_labels = subset.value[:, 5:]
        test_inputs = subset_test_list[index].value[:, 0:5]
        test_labels = subset_test_list[index].value[:, 5:]
        
        train(net, training_inputs, training_labels, EPOCHS, LR, bs)
        avg_mse = test(test_inputs, test_labels, net)
        MSEs.append(avg_mse)

    avg_mse = sum(MSEs)/len(MSEs)
    MODELS['{a}_{x}-{y}_{z}_{b}_{c}'.format(a=HL, x=HN1, y=HN2, z=EPOCHS, b=LR, c=bs)] = avg_mse

with open('Data/Search/k_fold_results_{x}HL_bs.csv'.format(x=HL), 'w') as f:
    for key in MODELS.keys():
        f.write("%s: %s\n"%(key, MODELS[key]))

print(MODELS)

# Should the test set be standardised as part of the training data?
# Or should all of the test data in the folds be standardised together - excluding the train data?
# Or can I standardise before splitting into folds and replication?
# Ask Gabe what he did for manual search standardisation
# Right now i will standardise at the outset
# In manual search i standardised after replication
# worst case: test both
# If i standardise at the outset, I have to add the rates of change to the standardised inputs and subsequently unstandaridse the output. 

# Split Training Data into Inputs and labels


# training_inputs = training_data_array[:, 0:5]
# training_labels = training_data_array[:, 5:]
# print(training_inputs)
# print(training_labels)

# standardised_inputs = preprocessing.scale(training_inputs)
# standardised_labels = preprocessing.scale(training_labels)

# print(standardised_inputs)
# print(standardised_labels)

# load traindata - decide on traindata based on advice from articles **
# convert pd dataframe to numpy array **
# Standardise Training Data **
# split data into k folds (k=6) **
# For each subset of data , replicate the training data (k-1)**
# Then calculate labels**
# Then remove 144 h input points**
# The convert subset_train_list and subset_test_list into numpy arrays**
# Shuffle data**
# standardise training data
# Implement training loop (5-3 neurons)