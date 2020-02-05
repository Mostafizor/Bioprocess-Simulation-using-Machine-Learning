import pandas as pd
import numpy as np 
from ann import Net
from replicate import replicate_data
from sklearn import preprocessing 
from sklearn.model_selection import KFold

# Load training data as pd dataframe and convert pd dataframe into numpy array.
training_data = pd.read_excel('Data/reduced_training_data.xlsx')
training_data_array = np.array(training_data)

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


# Standardise Training Data and Test Data
for subset in subset_train_list:
    subset.value = preprocessing.scale(subset.value)

for subset in subset_test_list:
    subset.value = preprocessing.scale(subset.value)    # STD of LI and NIC is zero: what to do?

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
# split data into k folds (k=6) **
# For each subset of data , replicate the training data (k-1)**
# Then calculate labels**
# Then remove 144 h input points**
# The convert subset_train_list and subset_test_list into numpy arrays**
# Shuffle data**
# standardise training data **
# split training data into inputs and labels
# Now that you have subsets, no need to perfrom this preprocessing procedure again; can now validate all hyperparameters
# Using a for loop, perform k-fold cross validation for each hyperparameter combination : EPOCHS and HN
# Calcualte average MSE for each subset and the the average MSE over all subsets.
# Store average MSE in a dictionary with the key indicating the network config
# Find the lowest average MSE
# test this model against testset and plot using matplotlib
