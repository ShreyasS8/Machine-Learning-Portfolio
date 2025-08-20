#### ASSIGNMENT1.2 PARTA AND PARTB
import sys
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

# SOFTMAX FUNCTION 
# INPUT: Z = X.W 
# OUTPUT: EXP(WT.X)/SUM OF EXP(WT.X)
def softmax_function(z):
    exp_of_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    g_of_z = exp_of_z / np.sum(exp_of_z, axis=1, keepdims=True)
    return g_of_z

# Y_PRED CALCULATOR
# INPUT: X,W
# OUTPUT: Z = SOFTMAX(X.W)
def y_predicted_train_complete(x, w):
    z_value_pred = np.dot(x, w)
    y_predicted_train_complete_data = softmax_function(z_value_pred)
    return y_predicted_train_complete_data

# GRADIENT CALCULATOR
# INPUT: X,Y_PRED,Y,FREQJ,N
# OUTPUT: GRADIENT OF LOSS
def gradient_G(x, y_predicted, y, frequencyj, N):
    #frequency_j_class = frequencyj[np.argmax(y, axis=1)][:, np.newaxis]
    gradient = np.dot(x.T, (y_predicted - y) / frequencyj) / (2 * N)
    return gradient

# LOSS FUNCTION
# INPUT: X,Y,W,FREQ
# OUTPUT: LOSS for MULTICLASS SOFTMAX CLASSIFIER 
def loss_function(x, y_data_train, w, frequencyj):
    n_value = y_data_train.shape[0]
    #z_value_pred = np.dot(x, w)
    y_predicted_train_complete_data = y_predicted_train_complete(x, w)  # softmax_function(z_value_pred)
    log_y_predicted = np.log(y_predicted_train_complete_data+1e-12)
    loss_funct_equ = - np.sum((y_data_train / frequencyj[:, np.newaxis].T) * log_y_predicted) / (2 * n_value)
    return loss_funct_equ

# CLASS OUTPUT FUNCTION
# INPUT: X,W
# OUTPUT: MULTICLASS INDEX
def predicted_values(x_test, weights_optimized):
    x_test_zero = np.ones((x_test.shape[0], 1))
    x_test_b = np.hstack((x_test_zero, x_test))
    z_test = np.dot(x_test_b, weights_optimized)
    y_test_predicted = softmax_function(z_test)
    return np.argmax(y_test_predicted, axis=1) + 1

# BINARY ENCODER FOR EACH CLASS FUNCTION
# INPUT: ACTUAL CLASS LIST,NO. OF CLASSES
# OUTPUT: Y (N * K) 
def ohe_y_values(y_data, n_class):
    y_ohe = np.eye(n_class)[y_data - 1]
    return y_ohe

# PROBABILITY PJ
# INPUT: X,W
# OUTPUT: PJ MATRIX
def probabilities_pj(x_test, weights_optimized):
    # x_test_zero = np.ones((x_test.shape[0], 1))
    # x_test_b = np.hstack((x_test_zero, x_test))
    z_test = np.dot(x_test, weights_optimized)
    y_test_predicted = softmax_function(z_test)
    return y_test_predicted

### LEARNING FUNCTION
# CONSTANT
def constant_learning_rate(const):
    return const
# ADAPTIVE ETA=ETA0/(1+K*t)
def adaptive_learning_rate(initial_learning_rate_o,k_value,t):
    adaptive_learning_rate_t = initial_learning_rate_o / (1 + (k_value * t))
    # print(f'Adaptive learning rate for epoch{t} is: ',adaptive_learning_rate_t)
    return adaptive_learning_rate_t
# TERNARY SEARCH FOR BEST ETA
def ternary_search_LR(x, y_data_train, frequencyj, N_complete, w,etah):
    etal = 0
    # etah = 1e-9
    y_predicted = y_predicted_train_complete(x, w)
    frequency_j_class = frequencyj[np.argmax(y_data_train, axis=1)][:, np.newaxis]
    g = gradient_G(x, y_predicted, y_data_train, frequency_j_class, N_complete)

    while loss_function(x, y_data_train, w, frequencyj) > loss_function(x, y_data_train, w - (etah * g), frequencyj):
        etah = 2 * etah

    for iteration in range(20):

        eta1 = (2 * etal + etah) / 3
        eta2 = (etal + 2 * etah) / 3

        if loss_function(x, y_data_train, (w - (eta1 * g)), frequencyj) < loss_function(x, y_data_train, (w - (eta2 * g)),
                                                                                      frequencyj):
            etah = eta2

        elif loss_function(x, y_data_train, (w - (eta1 * g)), frequencyj) > loss_function(x, y_data_train, (w - (eta2 * g)),
                                                                                        frequencyj):
            etal = eta1

        else:
            etal = eta1
            etah = eta2

    eta_final = (etal + etah) / 2
    return eta_final

from math import sqrt
def golden_section_search(x, y_data_train, frequencyj, N_complete, w,etal, etah, tol=1e-3):
    # GRADIENT CALCULATION
    y_predicted = y_predicted_train_complete(x, w)
    frequency_j_class = frequencyj[np.argmax(y_data_train, axis=1)][:, np.newaxis]
    g = gradient_G(x, y_predicted, y_data_train, frequency_j_class, N_complete)
    # CALCULATE EXTREME END OF ETA
    while loss_function(x, y_data_train, w, frequencyj) > loss_function(x, y_data_train, w - (etah * g), frequencyj):
        etah = 2 * etah
    # GOLDEN RATIO
    phi = (1 + sqrt(5)) / 2
    # POINTS eta1 AND eta2 BETWEEN etal AND etah
    eta1 = etah - (etah - etal) / phi
    eta2 = etal + (etah - etal) / phi
    # EVALUATE LOSS FUNCTION AT eta1 and eta2 
    loss_eta1 = loss_function(x, y_data_train, (w - (eta1 * g)), frequencyj)
    loss_eta2 = loss_function(x, y_data_train, (w - (eta2 * g)), frequencyj)
    # CONVERGENCE CRITERIA
    while (etah - etal) > tol:
        if loss_eta1 < loss_eta2:
            etah = eta2
            eta2 = eta1
            loss_eta2 = loss_eta1
            eta1 = etah - (etah - etal) / phi
            loss_eta1 = loss_function(x, y_data_train, (w - (eta1 * g)), frequencyj)
        else:
            etal = eta1
            eta1 = eta2
            loss_eta1 = loss_eta2
            eta2 = etal + (etah - etal) / phi
            loss_eta2 = loss_function(x, y_data_train, (w - (eta2 * g)), frequencyj)
    # MIDPOINT OF etal AND etah
    return (etal+etah)/2

### PARTA 
def PARTA():
    #python3 logistic.py a train1.csv params.txt modelweights.txt
    FILE_PATH_TRAIN=sys.argv[2]
    FILE_PATH_PARAMS=sys.argv[3]
    FILE_PATH_WEIGHTS=sys.argv[4]

    ### INPUT TRAINING DATA
    training_data = pd.read_csv(FILE_PATH_TRAIN)
    # ARRANGING INPUTS IN PROPER FORMAT
    x_train_data = training_data.iloc[:, :-1].values
    y_train_values = training_data.iloc[:, -1].values
    # ADDING ONES FOR INTERCEPT
    x_train_data = np.hstack([np.ones((x_train_data.shape[0], 1)), x_train_data])
    # CALCULATE FREQUENCY
    num_class = 4
    frequency_j = np.bincount(y_train_values, minlength=num_class)[1:]
    # INITIALIZING WEIGHT VECTOR
    m = x_train_data.shape[1]
    n = x_train_data.shape[0]
    weights_W = np.zeros((m, num_class))
    # ONE HOT ENCODING OF OUTPUT
    y_train_vales_OHE = ohe_y_values(y_train_values, num_class)
    
    ### ENTER BATCH SIZE,EPOCH SIZE & LEARNING STRATEGY
    with open(FILE_PATH_PARAMS, 'r') as file:
        params = file.read().strip().split('\n')
    num_epochs = int(params[2])
    batch_size_N = int(params[3])  # int(input("What is the batch size?: "))
    if(params[0]=='2'):
        parts = params[1].split(',')
        learn_rate = float(parts[0])
        k_value = float(parts[1])
    else:
        learn_rate = float(params[1])
    # EPOCH SIZE
    for num in range(num_epochs):
        # BATCH SIZE
        for last_batch_index in range(0, n, batch_size_N):
            if last_batch_index + batch_size_N > x_train_data.shape[0]:
                x_batch_data = x_train_data[last_batch_index:]
                y_batch_data = y_train_vales_OHE[last_batch_index:]
            else:
                x_batch_data = x_train_data[last_batch_index:last_batch_index + batch_size_N]
                y_batch_data = y_train_vales_OHE[last_batch_index:last_batch_index + batch_size_N]
    
            # CALCULATE Z=X.W
            #z_value = np.dot(x_batch_data, weights_W)
            y_predicted_train = y_predicted_train_complete(x_batch_data, weights_W)  # softmax_function(z_value)
            frequency_j_batch = frequency_j[np.argmax(y_batch_data, axis=1)][:, np.newaxis]
    
            # CALCULATE GRADIENT WITH COMPLETE CLASS FREQUENCY
            # grad_l = np.dot(x_batch_data.T, (y_predicted_train-y_batch_data) / frequency_j_batch) / (2*batch_size_N)
            grad_l = gradient_G(x_batch_data, y_predicted_train, y_batch_data, frequency_j_batch, batch_size_N)
            
            # CALCULATE LOSS
            #  z_value_pred = np.dot(x_train_data, weights_W)
            #  y_predicted_train_complete_data = softmax_function(z_value_pred)
            L_w = loss_function(x_train_data, y_train_vales_OHE, weights_W, frequency_j)
            
            # LEARNING RATE
            if(params[0]=='1'):
                a_learning_rate_adaptive = constant_learning_rate(learn_rate)
            if(params[0]=='2'):
                a_learning_rate_adaptive = adaptive_learning_rate(learn_rate,k_value,num+1)
            if(params[0]=='3'):
                a_learning_rate_adaptive = ternary_search_LR(x_batch_data, y_batch_data, frequency_j, y_batch_data.shape[0], weights_W,learn_rate)
                
            # UPDATING WEWIGHTS FOR THIS BATCH
            weights_W = weights_W - a_learning_rate_adaptive * grad_l
            
    # SAVING IN FILE_PATH_WEIGHTS
    np.savetxt(FILE_PATH_WEIGHTS, weights_W.reshape(-1, 1))
      
### PARTB
def PARTB():
    # python3 logistic.py b train1.csv test.csv modelweights.txt modelpredictions.csv
    FILE_PATH_TRAIN=sys.argv[2]
    FILE_PATH_TEST=sys.argv[3]
    FILE_PATH_WEIGHTS=sys.argv[4]
    FILE_PATH_PREDS=sys.argv[5]
    
    # FILE_PATH_TRAIN='train1.csv'
    # FILE_PATH_TEST='test1.csv'
    # FILE_PATH_WEIGHTS='modelweights.csv'
    # FILE_PATH_PREDS='modelpredictions.csv'
    
    # INPUT TRAINING DATA
    training_data = pd.read_csv(FILE_PATH_TRAIN)
    # ENTER INPUTS INTO PROPER FORMATS
    x_train_data = training_data.iloc[:, :-1].values
    y_train_values = training_data.iloc[:, -1].values
    # PRE PROCESSING
    scaler = preprocessing.StandardScaler().fit(x_train_data)
    x_train_data = scaler.transform(x_train_data)
    # ADDING ONES FOR INTERCEPT
    x_train_data = np.insert(x_train_data,0,np.ones(x_train_data.shape[0]),axis=1)   
    # FREQUENCY OF EACH CLASS
    num_class = 4
    frequency_j = np.bincount(y_train_values, minlength=num_class)[1:]
    # INITIALIZING WEIGHT VECTOR
    m = x_train_data.shape[1]
    n = x_train_data.shape[0]
    weights_W = np.zeros((m, num_class))

    # ONE HOT ENCODING OF OUTPUT
    y_train_vales_OHE = ohe_y_values(y_train_values, num_class)
    # LEARNING PARAMETERS IN GOLDEN SEARCH
    num_epochs = 7
    batch_size_N = x_train_data.shape[0]
    etal = 0
    etah = 1000
    tol=1e-1
    for num in range(num_epochs):
        for last_batch_index in range(0, n, batch_size_N):
            if last_batch_index + batch_size_N > x_train_data.shape[0]:
                x_batch_data = x_train_data[last_batch_index:]
                y_batch_data = y_train_vales_OHE[last_batch_index:]
            else:
                x_batch_data = x_train_data[last_batch_index:last_batch_index + batch_size_N]
                y_batch_data = y_train_vales_OHE[last_batch_index:last_batch_index + batch_size_N]

            # CALCULATE Z
            y_predicted_train = y_predicted_train_complete(x_batch_data, weights_W)
            frequency_j_batch = frequency_j[np.argmax(y_batch_data, axis=1)][:, np.newaxis]

            # CALCULATE GRADIENT WITH CLASS FREQUENCY
            grad_l = gradient_G(x_batch_data, y_predicted_train, y_batch_data, frequency_j_batch, batch_size_N)

            # CALCULATE LOSS FOR FULL BATCH
            # L_w = loss_function(x_train_data, y_train_vales_OHE, weights_W, frequency_j)

            # TERNARY SEARCH FOR OPTIMAL LEARNING RATE
            # tern_search_LR = ternary_search_LR(x_batch_data, y_batch_data, frequency_j, batch_size_N, weights_W, eta0)

            # GOLDEN SEARCH FOR OPTIMAL LEARNING RATE
            gold_search_LR = golden_section_search(x_batch_data, y_batch_data, frequency_j, batch_size_N, weights_W, etal,etah,tol)

            # WEIGHTS UPDATE
            # weights_W = weights_W - tern_search_LR * grad_l
            weights_W = weights_W - gold_search_LR * grad_l
            
    # SAVE WEIGHTS
    np.savetxt(FILE_PATH_WEIGHTS, weights_W.reshape(-1, 1))
    
    #  ENTER TEST DATA
    testing_data = pd.read_csv(FILE_PATH_TEST)
    x_test_data = testing_data.values
    # TEST DATA PRE PRCOSEESING
    x_test_data = scaler.transform(x_test_data)
    x_test_data = np.insert(x_test_data,0,np.ones(x_test_data.shape[0]),axis=1)
    # PROBABILITY MATRIX
    p_j = probabilities_pj(x_test_data, weights_W)
    # SAVE PROBABILITY MATRIX
    np.savetxt(FILE_PATH_PREDS, p_j, delimiter=',',fmt='%.20e')

# Checking for PART A or PART B
if(sys.argv[1]=='a'):
    PARTA()
elif(sys.argv[1]=='b'):
    PARTB()
else:
    print("Part A or B Not Specified Properly\n")