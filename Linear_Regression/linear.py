#### ASSIGNMENT1 PARTA AND PARTB
import sys
import os
import pandas as pd
import numpy as np
### PARTA 
def PARTA():
    FILE_PATH_TRAIN=sys.argv[2]
    FILE_PATH_TEST=sys.argv[3]
    FILE_PATH_WEIGHTS=sys.argv[4]
    FILE_PATH_MODEL_PRED=sys.argv[5]
    FILE_PATH_MODEL_WEIGHTS=sys.argv[6]
    ##########################  TRAINING DATA FOR LINEAR REGRESSION
    # Taking input from the CSV file into a matrix format X(n,m) 
    # Input vector has m features 
    # No of input vector are n 
    #FILE_PATH_TRAIN = r'D:\IIT DELHI M.Tech\ML COL774\Assignment1\train.csv'
    dataframe = pd.read_csv(FILE_PATH_TRAIN)
    
    # Exclude the last column ('Total Costs') for X
    df_features_X = dataframe.iloc[:, :-1]
    df_features_Y = dataframe.iloc[:, -1]

    # Convert the DataFrame to a NumPy array (matrix format)
    training_data_X = df_features_X.to_numpy()
    training_data_Y = df_features_Y.to_numpy()

    # Convert to Numpy format
    X_train = training_data_X
    Y_train = training_data_Y.reshape(-1,1)

    # Adding ones to the Matrix for b
    X_train_b = np.ones((X_train.shape[0], 1))
    X_train_b_added = np.hstack((X_train_b,X_train))
    
    # Inputting Sample Weights
    #FILE_PATH_WEIGHTS = r'D:\IIT DELHI M.Tech\ML COL774\Assignment1\sample_weights1.txt'
    sample_weights_U = np.loadtxt(FILE_PATH_WEIGHTS).reshape(-1,1)

    # Calculate XT.U separately and then XT.U.Y
    X_train_transpose = X_train_b_added.T
    X_transpose_U = X_train_transpose * sample_weights_U.reshape(-1)
    X_transpose_U_Y = X_transpose_U @ Y_train

    # Calculate inverse
    XT_U_X_inv = np.linalg.inv(X_transpose_U @ X_train_b_added)
    #XT_U_X_pseudo_inv = np.linalg.pinv(X_transpose_U @ X_train_b_added)
    
    # Calculating the best W values
    W_best = XT_U_X_inv @ X_transpose_U_Y
    # Writing into modelweights.txt
    np.savetxt(FILE_PATH_MODEL_WEIGHTS,W_best)
    
    ##########################  TESTING DATA
    # Taking predictions from the CSV file into a column vector  
    #FILE_PATH_TEST = r'D:\IIT DELHI M.Tech\ML COL774\Assignment1\test.csv'
    dataframe_test = pd.read_csv(FILE_PATH_TEST)

    # Inputting X Training
    X_test = dataframe_test.values

    # Adding ones to the Matrix for b
    X_test_b = np.ones((X_test.shape[0], 1))
    X_test_b_added = np.hstack((X_test_b,X_test))

    # Calculate Y^ = X*W_best
    Y_test = X_test_b_added @ W_best
    # Writing into modelpredictions.txt
    np.savetxt(FILE_PATH_MODEL_PRED,Y_test)
    #np.savetxt(FILE_PATH_MODEL_PRED, Y_test, fmt='%f', delimiter='\n')

### PARTA 
def PARTB():
    FILE_PATH_TRAIN=sys.argv[2]
    FILE_PATH_TEST=sys.argv[3]
    FILE_PATH_LAMBDA=sys.argv[4]
    FILE_PATH_MODEL_PRED=sys.argv[5]
    FILE_PATH_MODEL_WEIGHTS=sys.argv[6]
    FILE_PATH_BEST_LAMBA=sys.argv[7]
    ##########################  TRAINING DATA FOR RIGGE REGRESSION
    #FILE_PATH_TRAIN = 'train.csv'
    dataframe = pd.read_csv(FILE_PATH_TRAIN)

    # Exclude the last column ('Total Costs') for X
    df_features_X = dataframe.iloc[:, :-1]
    df_features_Y = dataframe.iloc[:, -1]

    # Convert the DataFrame to a NumPy array (matrix format)
    training_data_X = df_features_X.to_numpy()
    training_data_Y = df_features_Y.to_numpy()

    # Convert to Numpy format
    X_train = training_data_X
    Y_train = training_data_Y.reshape(-1,1)

    # Adding ones to the Matrix for b
    X_train_b = np.ones((X_train.shape[0], 1))
    X_train_b_added = np.hstack((X_train_b,X_train))
      
    # Taking Input the set of lambda values
    #FILE_PATH_LAMBDA = 'regularization.txt'
    lambda_para = np.loadtxt(FILE_PATH_LAMBDA)

    # Cross Validation
    number_fold = 10
    sample_index = number_fold*(X_train.shape[0]//number_fold)

    # Input Data
    X_ridge_reg_all = X_train[:sample_index]
    Y_ridge_reg = Y_train[:sample_index]

    # Adding ones for b to X
    X_ridge_reg = np.c_[np.ones((X_ridge_reg_all.shape[0], 1)), X_ridge_reg_all]

    # Sample index for splitting training data
    index_arr = np.arange(sample_index)

    # Number of samples in a single fold
    sample_size = sample_index//number_fold

    # Ridge Regression Parameters
    W_ridge_reg_best = None
    Y_predicted = None
    X_train_ridge = None
    Y_train_ridge = None

    # Storing MSE values
    lambda_mse_data = {}
    lambda_w_data = {}
    lambda_Y_ridge_pred = {}

    # 10-fold Cross Validation process
    for lam in lambda_para:
        mse_values = []
        for fold in range(0, number_fold):
            start_index = fold * sample_size
            end_index = sample_size + start_index

            # Dividing data in Test and Train set
            test_set = index_arr[start_index:end_index]
            training_set = np.concatenate([index_arr[:start_index], index_arr[end_index:]])
            
            # Train and Test data separated
            X_train_ridge, Y_train_ridge = X_ridge_reg[training_set], Y_ridge_reg[training_set]
            X_test_ridge, Y_test_ridge = X_ridge_reg[test_set], Y_ridge_reg[test_set]
            
            # Calculating W=inverse(XT.X + lambda*I).XT.Y
            a_part_W = np.dot(X_train_ridge.T, X_train_ridge) + lam * np.eye(X_train_ridge.shape[1])
            a_part_W_inv = np.linalg.inv(a_part_W)
            b_part_W = np.dot(X_train_ridge.T, Y_train_ridge)

            # W Best calculation
            W_ridge_reg_best = np.dot(a_part_W_inv, b_part_W)
            #np.savetxt(W_ridge_reg_best_file, W_ridge_reg_best)

            # Prediction from Ridge Reg
            Y_pred_ridge = np.dot(X_test_ridge, W_ridge_reg_best)

            # Calculate mean square error
            mean_sq_er = np.mean((Y_test_ridge - Y_pred_ridge)**2)
            mse_values.append(mean_sq_er)
        
        # SUM of MSE of each lambda
        sum_of_mse = np.sum(mse_values)
        lambda_mse_data[lam] = sum_of_mse

    # Best lambda
    best_lambda = min(lambda_mse_data,key=lambda_mse_data.get)
    # Open the file in write mode
    # Write the variable's value to the file
    with open(FILE_PATH_BEST_LAMBA, 'w') as file:
        file.write(str(best_lambda))
        
    # Calculate best W for Train Data
    a_part_W_final = np.dot(X_ridge_reg.T, X_ridge_reg) + best_lambda * np.eye(X_ridge_reg.shape[1])
    a_part_W_final_inv = np.linalg.inv(a_part_W_final)
    b_part_W_final = np.dot(X_ridge_reg.T, Y_ridge_reg)

    # Save in modelweights.txt
    W_retrain_final = np.dot(a_part_W_final_inv, b_part_W_final)
    np.savetxt(FILE_PATH_MODEL_WEIGHTS, W_retrain_final)

    ##########################  TESTING DATA FOR RIDGE REGRESSION
    # Taking predictions from the CSV file into a column vector  
    #FILE_PATH_TEST = r'D:\IIT DELHI M.Tech\ML COL774\Assignment1\test.csv'
    dataframe_test = pd.read_csv(FILE_PATH_TEST)

    # Inputting X Training
    X_test = dataframe_test.values

    # Adding ones to the Matrix for b
    X_test_b = np.ones((X_test.shape[0], 1))
    X_test_b_added = np.hstack((X_test_b,X_test))

    # Calculate Y^ = X*W_best
    Y_test = X_test_b_added @ W_retrain_final
    # Writing into modelpredictions.txt
    np.savetxt(FILE_PATH_MODEL_PRED,Y_test)

# Checking for PART A or PART B
if(sys.argv[1]=='a'):
    PARTA()
elif(sys.argv[1]=='b'):
    PARTB()
else:
    print("Part A or B Not Specified Properly\n")