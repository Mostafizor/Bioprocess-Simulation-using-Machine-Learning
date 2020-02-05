import pandas as pd
import numpy as np 
from ann import Net
from replicate import replicate_data
from sklearn import preprocessing 
from sklearn.model_selection import KFold

# Load training data as pd dataframe and convert pd dataframe into numpy array.
training_data = pd.read_excel('Data/reduced_training_data.xlsx')

# Replicate the training data
replicated_data1 = replicate_data(training_data, 50, 0.03)
replicated_data2 = replicate_data(training_data, 50, 0.03)

training_data = training_data.append(replicated_data1, ignore_index=True, sort=False)
training_data = training_data.append(replicated_data2, ignore_index=True, sort=False)

training_data = np.array(training_data)

# Calculate Training Labels
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

# Shuffle Training Data
np.random.shuffle(training_data)

# Standardise the Training Data
training_data = preprocessing.scale(training_data)  ## Could be an issue with scaling rates of change

# Split the data into k=10 folds
kf = KFold(n_splits=10)
kf.get_n_splits(training_data)

# Split training data set into 10 subsets containing k-1 folds before optimisation.
class wrapper(object):
    def __init__(self):
        self.value = []

subset_train1 = wrapper() 
subset_train2 = wrapper()
subset_train3 = wrapper()
subset_train4 = wrapper()
subset_train5 = wrapper()
subset_train6 = wrapper()
subset_train7 = wrapper()
subset_train8 = wrapper()
subset_train9 = wrapper()
subset_train10 = wrapper()
subset_test1 = wrapper() 
subset_test2 = wrapper()
subset_test3 = wrapper()
subset_test4 = wrapper()
subset_test5 = wrapper()
subset_test6 = wrapper()
subset_test7 = wrapper()
subset_test8 = wrapper()
subset_test9 = wrapper()
subset_test10 = wrapper()
subset_train_list = [subset_train1, subset_train2, subset_train3, subset_train4, subset_train5, subset_train6, subset_train7, subset_train8, subset_train9, subset_train10]
subset_test_list = [subset_test1, subset_test2, subset_test3, subset_test4, subset_test5, subset_test6, subset_test7, subset_test8, subset_test9, subset_test10]

index = 0
for train_index, test_index in kf.split(training_data):

    for row in train_index:
        subset_train_list[index].value.append(training_data[row])
    
    for row in test_index:
        subset_test_list[index].value.append(training_data[row])
    
    index +=1

print(len(subset_train1.value))
# df = pd.DataFrame(training_data)
# df.to_excel('Data/scaled_and_shuffled_training_data.xlsx')


# Load traindata **
# Convert pd dataframe to numpy array **
# Replicate the Data  **
# Convert replicated data into numpy array **
# Calcualte Labels **
# pop 144 h **
# Shuffle it **
# Standardise the data **
# Split the data in k=10 folds **
# split training data into inputs and labels
# Now that you have subsets, no need to perfrom this preprocessing procedure again; can now validate all hyperparameters
# Using a for loop, perform k-fold cross validation for each hyperparameter combination : EPOCHS and HN
# Calcualte average MSE for each subset and the the average MSE over all subsets.
# Store average MSE in a dictionary with the key indicating the network config
# Find the lowest average MSE
# test this model against testset and plot using matplotlib