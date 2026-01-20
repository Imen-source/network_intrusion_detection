import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path, column_names):
    """Load train and test CSV files"""
    train_df = pd.read_csv(train_path, names=column_names)
    test_df = pd.read_csv(test_path, names=column_names)
    return train_df, test_df

def clean_numeric(df, numeric_cols):
    """
    Convert numeric columns to float, replace non-numeric values with 0
    """
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert non-numeric to NaN
    df[numeric_cols] = df[numeric_cols].fillna(0)  # replace NaN with 0
    return df

def encode_categorical(train_df, test_df, categorical_cols):
    """
    One-hot encode categorical columns.
    Ensures train and test have same columns after encoding.
    """
    combined = pd.concat([train_df, test_df], axis=0)
    combined = pd.get_dummies(combined, columns=categorical_cols)

    # Split back into train and test
    train_df = combined.iloc[:len(train_df), :].reset_index(drop=True)
    test_df = combined.iloc[len(train_df):, :].reset_index(drop=True)

    return train_df, test_df

def scale_numeric(train_df, test_df, numeric_cols):
    """Scale numeric columns using StandardScaler"""
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_df, test_df
