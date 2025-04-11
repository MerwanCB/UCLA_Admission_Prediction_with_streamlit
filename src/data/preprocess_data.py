import pandas as pd
import os

def preprocess_target(df, target_col, threshold):
    """
    Convert a continuous target variable to binary based on a threshold.
    """
    if target_col in df.columns:
        df[target_col] = (df[target_col] >= threshold).astype(int)
        print(f"Target column '{target_col}' converted to binary using threshold {threshold}.")
    else:
        print(f"Warning: Target column '{target_col}' not found.")
    return df

def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame.
    """
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols, axis=1)
        print(f"Dropped columns: {existing_cols}")
    else:
        print("No specified columns found to drop.")
    return df

def change_dtype_to_object(df, columns_to_object):
    """
    Change the data type of specified columns to 'object'.
    """
    for col in columns_to_object:
        if col in df.columns:
            df[col] = df[col].astype('object')
            print(f"Changed dtype of column '{col}' to object.")
        else:
            print(f"Warning: Column '{col}' not found for dtype change.")
    return df

def save_processed_data(df, file_name, data_dir="data/processed"):
    """
    Save the processed DataFrame to a CSV file.
    """
    os.makedirs(data_dir, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(data_dir, file_name)
    try:
        df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")