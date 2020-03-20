import numpy as np
import pandas as pd

## Create a function which takes in a dataset and replicates it
def replicate_data(data, replications, noise):                                                      # Create a function which accepts a dataset and replicates it
    cols = list(data.columns) 
    dataR = data[cols[0:3]]                                                                         # Create a secondary dataframe containing only columns 1-3 (the columns we want to replicate)
    df = data                                                                                       # Create the output dataframe that will contain both the original and the replicated data
    new_data = pd.DataFrame(columns=data.columns)
    i = 0                                                                                           # Initialise replication counter to 0
    while i < replications:
        replicated_data =  np.random.uniform(dataR-dataR*noise, dataR+dataR*noise)                  # Create random noise for each value in columns 2-4 of dataset
        replicated_data = pd.DataFrame(data=replicated_data, index=None, columns=dataR.columns)      # Cast the replicated data as a pandas DataFrame Object
        replicated_data['LI'] = df[cols[3]]                                                          # Add the missing light intensity column back into the replicated_data set
        replicated_data['NIC'] = df[cols[4]]                                                         # Add the missing nitrate inflow concentration back into the replicated_data set
        new_data = new_data.append(replicated_data, ignore_index=True, sort=False)
        i += 1
    return new_data

