# Bioprocess-Simulation-using-Machine-Learning
If you are coming from the GitHub link provided in my dissertation, the relevant folders are: 
1. FNN_Stopping_Grid 
2. RNN_Stopping_Grid

# Contents of FNN_Stopping_Grid Folder
* 2HL_manual_search_hn_e.py: optimises number of hidden neurons and EPOCHS of an FNN containing 2 hidden layers. The optimisation procedure utilised is 'Grid Search with Stopping Conditions' as outlined in the dissertation
* 2HL_manual_search_lr.py: optimises the learning rate using the the grid search approach as outlined in the dissertation
* 2HL_manual_search_bs.py: optimises the batch size using the the grid search approach as outlined in the dissertation
* model_train_test.py: used to train and test the optimised FNN

# Contents of RNN_Stopping_Grid Folder: 
* 1HL_manual_search_hn_e.py: optimises number of hidden neurons and EPOCHS of an RNN containing 1 hidden layer. The optimisation procedure utilised is also 'Grid Search with Stopping Conditions'
* 1HL_manual_search_lr.py: optimises the learning rate using the the grid search approach
* 1HL_manual_search_bs.py: optimises the batch size using the the grid search approach
* model_train_test.py: used to train and test the optimised RNN
