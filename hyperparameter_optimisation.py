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

    subset.value = np.array(df)

# Calculate training and test labels
for index1, subset in enumerate(subset_train_list):
    a = []
    
    try:
        for index2, row in enumerate(subset.value):
            dBC = subset_train_list[index1].value[index2 + 1][0] - row[0]
            dNC = subset_train_list[index1].value[index2 + 1][1] - row[1]
            dLP = subset_train_list[index1].value[index2 + 1][2] - row[2]

            rates =[dBC, dNC, dLP]
            a.append(rates)
    except IndexError:
        rates = [0, 0, 0]
        a.append(rates)
    
    a = np.array(a)
    subset_train_list[index1].value = np.append(subset_train_list[index1].value, a, axis=1)

for index1, subset in enumerate(subset_test_list):
    b = []
    
    try:
        for index2, row in enumerate(subset.value):
            dBC = subset_test_list[index1].value[index2 + 1][0] - row[0]
            dNC = subset_test_list[index1].value[index2 + 1][1] - row[1]
            dLP = subset_test_list[index1].value[index2 + 1][2] - row[2]

            rates =[dBC, dNC, dLP]
            b.append(rates)
    except IndexError:
        rates = [0, 0, 0]
        b.append(rates)
    
    b = np.array(b)
    subset_test_list[index1].value = np.append(subset_test_list[index1].value, b, axis=1)

print(subset_test2.value)

# for index1, subset in enumerate(subset_train_list):
#     a = np.array([[]])
#     try:
#         for index2, row in enumerate(subset.value):
#             dBC = subset_train_list[index1].value[index2 + 1][0] - subset_train_list[index1].value[index2][0]
#             dNC = subset_train_list[index1].value[index2 + 1][1] - subset_train_list[index1].value[index2][1]
#             dLP = subset_train_list[index1].value[index2 + 1][2] - subset_train_list[index1].value[index2][2]

#             a = np.append(a, [[dBC, dNC, dLP]], axis=1)
#             print(a)
#         subset_train_list[index1].value = np.append(subset_train_list[index1].value, a, axis=1)
#     except IndexError:
#         (subset_train_list[index1].value)[index2][5:8] = 0

#print(subset_train1.value)

# for index1, subset in enumerate(subset_train_list):
#     prevRow = []
#     prevSubset = []
#     for index2, row in enumerate(subset.value):
#         if index2 != 0:
#             dBC = row[0] - prevRow[0]
#             dNC = row[1] - prevRow[1]
#             dLP = row[2] - prevRow[2]
#             subset_train_list[index1][row].append(dBC)
#             subset_train_list[index1][row].append(dNC)
#             subset_train_list[index1][row].append(dLP)
#             prevRow = row
#         elif index2 == 0:
#             subset_train_list[index1][index2+1] - subset_train_list[index1][index2]


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
# Then remove 144 h input points
# standardise replicated training data
# Shuffle data
# standardise the isolated validation set
# Now that you have subsets, no need to perfrom this preprocessing procedure again; can now validate all hyperparameters
# Using a for loop, perform k-fold cross validation for each hyperparameter combination : EPOCHS and HN
# Calcualte average MSE for each subset and the the average MSE over all subsets.
# Store average MSE in a dictionary with the key indicating the network config
# Find the lowest average MSE
# test this model against testset and plot using matplotlib
