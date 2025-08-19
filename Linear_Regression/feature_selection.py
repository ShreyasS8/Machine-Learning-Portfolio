import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import alpha
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LassoLarsCV

# python3 feature_selection.py train.csv created.txt selected.txt
FILE_PATH_TRAIN=sys.argv[1]
FILE_PATH_CREATION=sys.argv[2]
FILE_PATH_SELECTION=sys.argv[3]

#training_data = pd.read_csv(train_data_csv_path)
training_data = pd.read_csv(FILE_PATH_TRAIN)
x_train_data = training_data.iloc[:, :-1]
y_train_value = training_data.iloc[:, -1]

# product of two existing feature
x_train_data['APR MDC Description_Facility Name'] = x_train_data['APR MDC Description'] * x_train_data['Facility Name']
x_train_data['APR MEDICAL_CCSR CODE'] = x_train_data['APR Medical Surgical Description'] * x_train_data['CCSR Diagnosis Code']
x_train_data['Permanent Facility Id_Facility_name'] = x_train_data['Permanent Facility Id'] * x_train_data[
    'Facility Name']

# ratio
x_train_data['Permanent Facility Id_Facility_name'] = x_train_data['Permanent Facility Id'] / (
            x_train_data['Facility Name'] + 1e-8)
x_train_data['APR MDC Description_DRG Code divide'] = x_train_data['APR MDC Description'] / x_train_data['APR DRG Code']

# Interaction feature
x_train_data['CCSR Diagnosis Code_APR MDC Code'] = x_train_data['CCSR Procedure Code'] + x_train_data['APR MDC Code']
x_train_data['APR MDC Description_DRG Code int'] = x_train_data['APR MDC Description'] + x_train_data['APR DRG Code']
x_train_data['Payment Typology 3 squared'] = x_train_data['Payment Typology 3'] ** 2

# Polynomial
x_train_data['APR Severity of Illness Code squared'] = x_train_data['APR Severity of Illness Code'] ** 2

x_train_data['CCSR Procedure Description 1'] = x_train_data['CCSR Procedure Description'] ** 2

# One Hot encoding
feature_column_OHE_array = ['Patient Disposition', 'APR Medical Surgical Description', 'Age Group', 'Gender', 'Emergency Department Indicator', 'APR Risk of Mortality' ,'APR Severity of Illness Description']
OHE_data = OneHotEncoder(sparse_output=False, drop='first')
feature_column_OHE = feature_column_OHE_array

# One Hot encoding on training data
OHE_columns_added_train = OHE_data.fit_transform(x_train_data[feature_column_OHE])
OHE_data_frame_train = pd.DataFrame(OHE_columns_added_train, columns=OHE_data.get_feature_names_out(feature_column_OHE))

x_combined_train_OHE_data = pd.concat([x_train_data.drop(columns=feature_column_OHE), OHE_data_frame_train], axis=1)


# Lasso model fit with The Least Angle Regression (Lars)
lasso_lars_cross_valid = LassoLarsCV(cv=10)
lasso_lars_cross_valid.fit(x_combined_train_OHE_data, y_train_value)
#best_lamda_cv = lasso_lars_cross_valid.alpha_
#alpha_shape=lasso_lars_cross_valid.alphas_

# Feature selection in training based on Lasso Lars Regression
feature_selection_lars = lasso_lars_cross_valid.coef_ != 0
x_lars_data_train = x_combined_train_OHE_data.loc[:, feature_selection_lars]

headers_created = x_combined_train_OHE_data.columns.tolist()
# Creation of created.txt and selected.txt
x_lars_data_train_df = pd.DataFrame(x_lars_data_train)
headers_selected = x_lars_data_train_df.columns.tolist()


required_col_name = 'Birth Weight'
required_index = headers_created.index(required_col_name)
columns_after_required = headers_created[required_index+1:]

with open(FILE_PATH_CREATION, 'w') as file:
    for column in columns_after_required:
        file.write(column + '\n')

required_index_s = headers_selected.index(required_col_name)
column_present = headers_selected[required_index_s+1:]

with open(FILE_PATH_SELECTION, 'w') as file:
    for column_selected in columns_after_required:
        if column_selected in column_present:
            file.write('1\n')
        if column_selected not in column_present:
            file.write('0\n')