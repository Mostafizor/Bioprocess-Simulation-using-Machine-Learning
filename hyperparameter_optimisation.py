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

# Replicate the training and testing data in each subset.
columns = "BC NC LP LI NIC".split()
for index, subset in enumerate(subset_train_list):
    df = pd.DataFrame(data=subset.value, index=None, columns=columns)
    ref = df

    replicated_data1 = replicate_data(ref, 50, 0.03)
    df = df.append(replicated_data1, ignore_index=True, sort=False)

    replicated_data2 = replicate_data(ref, 50, 0.05)
    df = df.append(replicated_data2, ignore_index=True, sort=False)

    subset.value = df

print(subset_train1.value, subset_train2.value, subset_train6.value)

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
# Then calculate labels
# standardise replicated training data
# Shuffle data
# standardise the isolated validation set
# Now that you have subsets, no need to perfrom this preprocessing procedure again; can now validate all hyperparameters
# Using a for loop, perform k-fold cross validation for each hyperparameter combination : EPOCHS and HN
# Calcualte average MSE for each subset and the the average MSE over all subsets.
# Store average MSE in a dictionary with the key indicating the network config
# Find the lowest average MSE
# test this model against testset and plot using matplotlib 
