# ASSIGNMENT 1.2 PARTC
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

# python3 feature_selection.py train.csv created.txt selected.txt
# FILE PATH
FILE_PATH_TRAIN=sys.argv[1]
FILE_PATH_CREATION=sys.argv[2]
FILE_PATH_SELECTION=sys.argv[3]

# JSON READING
with open(r'mapping.json','r') as json_file:
    data_dict = json.load(json_file)

### TRAIN DATA
target = "Gender"
x_train = pd.read_csv(FILE_PATH_TRAIN)
for key in data_dict.keys():
    #print(key)
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

# TRAIN DATA
training_data = pd.read_csv(FILE_PATH_TRAIN)
x_train_data = x_train.iloc[:, :-1]
y_train_values = training_data.iloc[:, -1]
# TRANSFORM LABELS
y_train_values = (y_train_values+1) //2 + 1

# FINDING COLUMNS
columns_after = set(x_train_data)
# FEATURE SELECTION
significant_features = anova_feature_selection(x_train_data, y_train_values)
x_train_data_1 = x_train_data.iloc[:, significant_features]

x_data = training_data.iloc[:, -1]
x_data = pd.DataFrame(x_data)
columns_before = set(x_data.columns)
# CREATION FILE
columns_created = columns_after - columns_before

with open(FILE_PATH_CREATION, 'w') as file:
    for column in columns_created:
        file.write(column + '\n')

after_selection = pd.DataFrame(x_train_data_1)
columns_selected = set(after_selection.columns)

# final_columns_selected = columns_created - columns_selected
# SELECTION FILE
with open(FILE_PATH_SELECTION, 'w') as file:
    for column_selected in columns_created:
        if column_selected in columns_selected:
            file.write('1\n')
        if column_selected not in columns_selected:
            file.write('0\n')