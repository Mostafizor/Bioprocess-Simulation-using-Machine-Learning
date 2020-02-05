import pandas as pd 
from sklearn import preprocessing 

X = pd.read_excel('Data/standardise_test.xlsx')
y = preprocessing.scale(X)

standardised_data = pd.DataFrame(y)
standardised_data.to_excel('Data/hehe.xlsx')