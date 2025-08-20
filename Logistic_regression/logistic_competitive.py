# ASSIGNMENT 1.2 PARTC
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

# python3 logistic competitive.py train2.csv test.csv output.txt
# FILE PATH
FILE_PATH_TRAIN=sys.argv[1]
FILE_PATH_TEST=sys.argv[2]
FILE_PATH_PREDS=sys.argv[3]

# JSON READING
with open(r'Data\mapping.json','r') as json_file:
    data_dict = json.load(json_file)

# OHD USING MAPPING JSON
target = "Gender"
x_train = pd.read_csv(FILE_PATH_TRAIN)
for key in data_dict.keys():
    # print(key)
    if key==target:
        continue
    possible_values = data_dict[key]
    one_hot = pd.get_dummies(x_train[key])
    one_hot = one_hot.reindex(columns=sorted(possible_values),fill_value=False)
    one_hot = one_hot.iloc[:,1:]
    one_hot = one_hot.astype(int)
    one_hot.columns = [f"{key}_{col}" for col in one_hot.columns]
    one_hot_array = one_hot.values
    column_index = x_train.columns.get_loc(key)
    x_train = x_train.drop(columns=[key])
    df_array = x_train.values
    columns_before = df_array[:, :column_index]
    columns_after = df_array[:, column_index:]
    new_columns = list(x_train.columns[:column_index]) + list(one_hot.columns) + list(x_train.columns[column_index:])
    combined_array = np.hstack([df_array[:, :column_index], one_hot_array, df_array[:, column_index:]])
    x_train = pd.DataFrame(combined_array, columns=new_columns)

# SIGMOID FUNCTION 
# INPUT: Z = X.W 
# OUTPUT: EXP(WT.X)/1+EXP(WT.X)
def sigmoid_function(z):
    exp_of_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    g_of_z = exp_of_z / np.sum(exp_of_z, axis=1, keepdims=True)
    return g_of_z

# PROBABILITY PJ
# INPUT: X,W
# OUTPUT: PJ MATRIX
def probabilities_pj(x_test, weights_optimized):
    z_test_1 = np.dot(x_test, weights_optimized)
   # z_test_2 = np.dot(x_test, weights_optimized)
    y_test_predicted = sigmoid_function(z_test_1)
    return y_test_predicted

# Y_PRED CALCULATOR
# INPUT: X,W
# OUTPUT: Z = SIGMOID(X.W)
def y_predicted_train_complete(x, w):
    z1 = np.dot(x, w)
    #z2 = np.dot(x, w[:,1])
    y_predicted_train_complete_data = sigmoid_function(z1)
    return y_predicted_train_complete_data

# GRADIENT CALCULATOR
# INPUT: X,Y_PRED,Y,FREQJ,N
# OUTPUT: GRADIENT OF LOSS
def gradient_G(x, y_predicted, y, frequencyj, N):
    frequency_j_class = frequencyj[np.argmax(y, axis=1)][:, np.newaxis]
    gradient = np.dot(x.T, (y_predicted - y) / frequency_j_class) /  (2 * N)
    return gradient

# LOSS FUNCTION
# INPUT: X,Y,W,FREQ
# OUTPUT: LOSS for BINARY SOFTMAX CLASSIFIER 
def loss_function(x, y_data_train, w, frequencyj):
    n_value = y_data_train.shape[0]
    #z_value_pred = np.dot(x, w)
    y_predicted_train_complete_data = y_predicted_train_complete(x, w)  # sigmoid_function(z_value_pred)
    log_y_predicted = np.log(y_predicted_train_complete_data)
    loss_funct_equ = - np.sum((y_data_train / frequencyj[:, np.newaxis].T) * log_y_predicted) / (2 * n_value)
    return loss_funct_equ

# CLASS OUTPUT FUNCTION
# INPUT: X,W
# OUTPUT: BINARY INDEX
def predicted_values(x_test, weights_optimized):
    x_test_zero = np.ones((x_test.shape[0], 1))
    x_test_b = np.hstack((x_test_zero, x_test))
    z_test_1 = np.dot(x_test_b, weights_optimized)
    #z_test_2 = np.dot(x_test_b, weights_optimized[:, 1])
    y_test_predicted = sigmoid_function(z_test_1)
    return np.argmax(y_test_predicted, axis=1) + 1

# FEATURE SELECTION
def anova_feature_selection(x, y, k=900):
    f_values, p_values = f_classif(x, y)
    significant_feat = np.where(p_values<0.09)[0]

    if len(significant_feat) == 0:
        # print("No feature meet p_value threshold")
        return 0

    if k is not None:
        signi_feat = significant_feat[np.argsort(f_values[significant_feat])[-k:]]

    else:
        signi_feat = significant_feat

    return signi_feat

# BINARY ENCODER FOR EACH CLASS FUNCTION
# INPUT: ACTUAL CLASS LIST,NO. OF CLASSES
# OUTPUT: Y (N * K) 
def ohe_y_values(y_data, n_class):
    #y_data = y_data.astype(int)
    y_ohe = np.eye(n_class)[y_data-1]
    # print(y_ohe)
    return y_ohe

# PCA REDUCTION
def pca_reduction(x):
    k_value = 800
    pca = PCA(n_components = k_value)
    x = pca.fit_transform(x)
    return x

# TERNARY SEARCH FOR BEST ETA
def ternary_search_LR(x, y_data_train, frequencyj, N_complete, w, eta_ini):
    etal = 0
    etah = eta_ini
    y_predicted = y_predicted_train_complete(x, w)
    # frequency_j_class = frequencyj[np.argmax(y_data_train, axis=1)][:, np.newaxis]
    g = gradient_G(x, y_predicted, y_data_train, frequencyj, N_complete)

    while loss_function(x, y_data_train, w, frequencyj) > loss_function(x, y_data_train, w - (etah * g), frequencyj):
        etah = 2 * etah
    for iteration in range(20):

        eta1 = (2 * etal + etah) / 3
        eta2 = (etal + 2 * etah) / 3
        if loss_function(x, y_data_train, (w - (eta1 * g)), frequencyj) > loss_function(x, y_data_train,
                                                                                        (w - (eta2 * g)),
                                                                                        frequencyj):
            etal = eta1

        elif loss_function(x, y_data_train, (w - (eta1 * g)), frequencyj) < loss_function(x, y_data_train,
                                                                                          (w - (eta2 * g)),
                                                                                          frequencyj):
            etah = eta2

        else:
            etal = eta1
            etah = eta2


    eta_final = (etal + etah) / 2
    return eta_final

# GOLDEN SEARCH
from math import sqrt
def golden_section_search(x, y_data_train, frequencyj, N_complete, w,etal, etah, tol=1e-3):
    # GRADIENT CALCULATION
    y_predicted = y_predicted_train_complete(x, w)
    # frequency_j_class = frequencyj[np.argmax(y_data_train, axis=1)][:, np.newaxis]
    g = gradient_G(x, y_predicted, y_data_train, frequencyj, N_complete)
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


### INPUT TRAINING DATA
training_data = pd.read_csv(FILE_PATH_TRAIN)
x_train_data = x_train.iloc[:, :-1]
y_train_values = training_data.iloc[:, -1]
# TRANSFORM LABELS
y_train_values = (y_train_values+1) //2 + 1

# PRE PROCESSING
scaler = preprocessing.StandardScaler().fit(x_train_data)
x_train_data = scaler.transform(x_train_data)
x_train_data = np.insert(x_train_data,0,np.ones(x_train_data.shape[0]),axis=1)

# FEATURE SELECTION
significant_features = anova_feature_selection(x_train_data, y_train_values)
x_train_data = x_train_data[:, significant_features]

# ONE HOT ENCODING OF OUTPUT
y_train_values = y_train_values.astype(int)
num_class = 2
frequency_j = np.bincount(y_train_values, minlength=num_class)[1:]

# INITIALIZING WEIGHT VECTOR
m = x_train_data.shape[1]
n = x_train_data.shape[0]
weights_W = np.zeros((m, num_class))

# LEARNING PARAMETERS IN TERNARY SEARCH
num_epochs = 10
batch_size_N = x_train_data.shape[0] # int(input("What is the batch size?: "))
etal = 0
etah = 1000
tol=1e-1
y_train_vales_OHE = ohe_y_values(y_train_values, num_class)

for num in range(num_epochs):
    for last_batch_index in range(0, n, batch_size_N):
        if last_batch_index + batch_size_N > x_train_data.shape[0]:
            x_batch_data = x_train_data[last_batch_index:]
            y_batch_data = y_train_vales_OHE[last_batch_index:]
        else:
            x_batch_data = x_train_data[last_batch_index:last_batch_index + batch_size_N]
            y_batch_data = y_train_vales_OHE[last_batch_index:last_batch_index + batch_size_N]

        # # CALCULATE Z
        y_predicted_train = y_predicted_train_complete(x_batch_data, weights_W)
        # frequency_j = frequency_j[np.argmax(y_batch_data, axis=1)][:, np.newaxis]

        # CALCULATE GRADIENT WITH CLASS FREQUENCY
        grad_l = gradient_G(x_batch_data, y_predicted_train, y_batch_data, frequency_j, batch_size_N)

        # Calculate loss for full data
        # L_w = loss_function(x_train_data, y_train_vales_OHE, weights_W, frequency_j)

        # TERNARY SEARCH FOR OPTIMAL LEARNING RATE
        # tern_search_LR = ternary_search_LR(x_batch_data, y_batch_data, frequency_j, batch_size_N, weights_W, eta0)
        
        # GOLDEN SEARCH FOR OPTIMAL LEARNING RATE
        gold_search_LR = golden_section_search(x_batch_data, y_batch_data, frequency_j, batch_size_N, weights_W, etal,etah,tol)

        # WEIGHTS UPDATE
        # weights_W = weights_W - tern_search_LR * grad_l
        weights_W = weights_W - gold_search_LR * grad_l
        

# SAVE WEIGHTS
# np.savetxt(FILE_PATH_WEIGHTS, weights_W.reshape(-1, 1))
# ##################################################################################################
# TEST DATA
target = "Gender"
x_test = pd.read_csv(FILE_PATH_TEST)
for key in data_dict.keys():
    #print(key)
    if key == target:
        continue
    possible_values = data_dict[key]
    one_hot = pd.get_dummies(x_test[key])
    one_hot = one_hot.reindex(columns=sorted(possible_values), fill_value=False)
    one_hot = one_hot.iloc[:, 1:]
    one_hot = one_hot.astype(int)
    one_hot.columns = [f"{key}_{col}" for col in one_hot.columns]
    one_hot_array = one_hot.values
    column_index = x_test.columns.get_loc(key)
    x_test = x_test.drop(columns=[key])
    df_array = x_test.values
    columns_before = df_array[:, :column_index]
    columns_after = df_array[:, column_index:]
    new_columns = list(x_test.columns[:column_index]) + list(one_hot.columns) + list(x_test.columns[column_index:])
    combined_array = np.hstack([df_array[:, :column_index], one_hot_array, df_array[:, column_index:]])
    x_test = pd.DataFrame(combined_array, columns=new_columns)

#  ENTER TEST DATA
testing_data = pd.read_csv(FILE_PATH_TEST)
x_test_data = x_test.values
# TEST DATA PRE PRCOSEESING
x_test_data = scaler.transform(x_test_data)
x_test_data = np.insert(x_test_data,0,np.ones(x_test_data.shape[0]),axis=1)
# FEATURE SELECTION
x_test_data = x_test_data[:, significant_features]
# PROBABILITY MATRIX
p_j = probabilities_pj(x_test_data, weights_W)
# TRANSFORM LABELS
predictions = np.argmax(p_j, axis=1)
predictions = (2 * predictions) - 1
# SAVE OUTPUT LABELS
np.savetxt(FILE_PATH_PREDS, predictions, delimiter=',',fmt='%.20e')