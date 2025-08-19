import argparse
import os
import warnings

import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LassoLars, LassoLarsCV

# plotting libs (imported in original code; kept for parity)
import seaborn as sns
import matplotlib.pyplot as plt

# silence some pandas warnings that may appear during in-place operations
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Run linear model pipeline (Code 2 behavior) with argparse.")
    p.add_argument('--train',  '-tr', required=True, help='Path to training CSV (train.csv)')
    p.add_argument('--test',   '-te', required=True, help='Path to testing CSV (test.csv)')
    p.add_argument('--out_test',  '-ot', required=True, help='Path to save test predictions (txt)')
    p.add_argument('--out_train', '-or', required=False, default=None,
                   help='Optional path to save train predictions (txt). If not provided, no train preds saved.')
    return p.parse_args()


# ---------------------------
# Utility functions (same behavior as Code 2)
# ---------------------------
def iqr_filter_dataframe(df, columns_to_filter):
    """
    Apply IQR filtering similar to your original block.
    Note: this modifies a copy and returns it.
    """
    filtered_df = df.copy()
    for col_name in columns_to_filter:
        Q1 = df[col_name].quantile(0.25)
        Q3 = df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        # original code uses 1.38 multiplier for upper bound
        upper_bound = Q3 + 1.38 * IQR
        filtered_df = filtered_df[(filtered_df[col_name] >= lower_bound) & (filtered_df[col_name] <= upper_bound)]
    return filtered_df


def OHE(dataframe):
    """
    One-hot encode the specified list of categorical columns.
    This function fits an encoder on the provided dataframe and returns
    the dataframe with encoded columns appended (original categorical cols dropped).
    This matches Code 2 behavior (train/test encoders are separate).
    """
    df = dataframe.copy()
    columns_to_encode = [
        'Race', 'Gender', 'Hospital Service Area', 'Hospital County',
        'Age Group', 'Ethnicity', 'Zip Code - 3 digits',
        'Type of Admission', 'Patient Disposition', 'APR MDC Code', 'Payment Typology 3',
        'Payment Typology 2'
    ]

    # keep only columns that exist in df (avoid KeyError)
    columns_to_encode = [c for c in columns_to_encode if c in df.columns]

    if len(columns_to_encode) == 0:
        return df  # nothing to encode

    # Warning if missing values present (same message as your code)
    if df[columns_to_encode].isnull().any().any():
        print("Warning: There are missing values in the columns to encode. Please handle them if needed.")

    # Fit and transform
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[columns_to_encode])
    encoded_columns = encoder.get_feature_names_out(columns_to_encode)

    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=df.index)

    df = df.drop(columns=columns_to_encode)
    df = pd.concat([df, encoded_df], axis=1)
    return df


def polyfeat(dataframe, columns):
    """
    For each column in 'columns', convert to numeric and apply PolynomialFeatures(degree=2).
    Concatenate generated polynomial feature columns into the dataframe.
    Matches Code 2 behavior.
    """
    df = dataframe.copy()
    poly = PolynomialFeatures(degree=2, include_bias=False)

    for column in columns:
        if column not in df.columns:
            # skip missing columns silently (keeps same behavior as original script which would raise)
            continue
        # coerce to numeric (original used errors='coerce')
        df[column] = pd.to_numeric(df[column], errors='coerce')
        # poly.fit_transform on single column returns two columns: [x, x^2]
        column_poly = poly.fit_transform(df[[column]])
        column_poly_df = pd.DataFrame(column_poly, columns=poly.get_feature_names_out([column]), index=df.index)
        df = pd.concat([df.reset_index(drop=True), column_poly_df.reset_index(drop=True)], axis=1)
    return df


def add_log_feature_at_end(dataframe, column_name):
    """
    Add np.log(column_name) as 'log_{column_name}' exactly as in your Code 2.
    This will produce -inf or NaN if values <= 0 — preserved to match original behavior.
    """
    df = dataframe
    if column_name not in df.columns:
        # if not present, simply return; original code would throw, but we keep it safe
        print(f"Warning: column '{column_name}' not present for log transformation.")
        return df
    # Note: this can produce warnings or -inf for non-positive numbers, matching original behavior.
    df[f'log_{column_name}'] = np.log(df[column_name])
    print("Shape after adding log feature:", df.shape)
    return df


def df_to_numpy_with_bias(df):
    """
    Convert dataframe-like to numpy array and add column of ones at beginning.
    Equivalent to the np.hstack((ones, df)) in your script but explicit and robust.
    """
    if isinstance(df, pd.DataFrame):
        arr = df.values
    else:
        arr = np.asarray(df)
    ones = np.ones((arr.shape[0], 1))
    stacked = np.hstack((ones, arr))
    return stacked


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    args = parse_args()

    FILE_PATH_TRAIN = args.train
    FILE_PATH_TEST = args.test
    FILE_PATH_PRED = args.out_test
    FILE_PATH_PRED_train = args.out_train

    # ---------------------------
    # Read training data
    # ---------------------------
    print("Reading training data from:", FILE_PATH_TRAIN)
    training_data = pd.read_csv(FILE_PATH_TRAIN)

    # ---------------------------
    # IQR filtering on 'Total Costs' (same as Code 2)
    # ---------------------------
    print("Applying IQR filter on 'Total Costs' (if present)...")
    if 'Total Costs' in training_data.columns:
        training_data = iqr_filter_dataframe(training_data, ['Total Costs'])
        print("Filtered training shape:", training_data.shape)
    else:
        print("Column 'Total Costs' not found in training data; skipping IQR filter.")

    # ---------------------------
    # Split X/y (assume last column is target as in original code)
    # ---------------------------
    x_train_data = training_data.iloc[:, :-1].copy()
    y_train_value = training_data.iloc[:, -1].copy()
    print("x_train shape:", x_train_data.shape, "y_train shape:", y_train_value.shape)

    # ---------------------------
    # One Hot Encoding (train)
    # ---------------------------
    print("Applying OneHotEncoder on training data...")
    x_train_data = OHE(x_train_data)

    # ---------------------------
    # Polynomial features (train)
    # ---------------------------
    poly_columns = ['APR Severity of Illness Code', 'CCSR Diagnosis Code',
                    'APR Risk of Mortality', 'APR MDC Description', 'Payment Typology 1']
    print("Applying polynomial features on columns:", poly_columns)
    x_train_data = polyfeat(x_train_data, poly_columns)

    # ---------------------------
    # Add log feature at end (train)
    # ---------------------------
    print("Adding log feature for 'Operating Certificate Number' (train)...")
    x_train_data = add_log_feature_at_end(x_train_data, 'Operating Certificate Number')

    # ---------------------------
    # Add bias column and convert to numpy
    # ---------------------------
    X_train_final = df_to_numpy_with_bias(x_train_data)
    print("Final X_train shape (with bias):", X_train_final.shape)

    # ---------------------------
    # Model training (Ridge alpha=2.0 as in Code 2)
    # ---------------------------
    print("Fitting Ridge(alpha=2.0) on training data...")
    model = Ridge(alpha=2.0)
    model.fit(X_train_final, y_train_value)
    print("Model fitted.")

    # ---------------------------
    # Testing data pipeline (mirrors train pipeline; NOTE: OHE fitted again separately)
    # ---------------------------
    print("Reading testing data from:", FILE_PATH_TEST)
    testing_data = pd.read_csv(FILE_PATH_TEST)
    x_test_data = testing_data.copy()

    print("Applying OneHotEncoder on testing data (separate fit, matching your original pipeline)...")
    x_test_data = OHE(x_test_data)

    print("Applying polynomial features on testing data...")
    x_test_data = polyfeat(x_test_data, poly_columns)

    print("Adding log feature for 'Operating Certificate Number' (test)...")
    x_test_data = add_log_feature_at_end(x_test_data, 'Operating Certificate Number')

    X_test_final = df_to_numpy_with_bias(x_test_data)
    print("Final X_test shape (with bias):", X_test_final.shape)

    # ---------------------------
    # Predictions and saving
    # ---------------------------
    print("Predicting on test data...")
    y_predicted_test = model.predict(X_test_final)
    np.savetxt(FILE_PATH_PRED, y_predicted_test)
    print(f"Saved {len(y_predicted_test)} test predictions to: {FILE_PATH_PRED}")

    if FILE_PATH_PRED_train:
        print("Predicting on training data (to save predictions to file)...")
        y_predicted_train = model.predict(X_train_final)
        np.savetxt(FILE_PATH_PRED_train, y_predicted_train)
        print(f"Saved {len(y_predicted_train)} train predictions to: {FILE_PATH_PRED_train}")

    print("Done.")


if __name__ == "__main__":
    main()
