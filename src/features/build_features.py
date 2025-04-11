import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def create_dummies(df, columns_to_dummy):
    """
    Create dummy variables for specified categorical columns.
    """
    # Filter valid object-type columns
    valid_columns = [col for col in columns_to_dummy if col in df.columns and df[col].dtype == 'object']
    if valid_columns:
        # Generate dummy variables
        df_dummies = pd.get_dummies(df, columns=valid_columns, dtype='int', drop_first=False)
        print(f"Created dummy variables for columns: {valid_columns}")
        return df_dummies
    else:
        print("No valid object type columns found to create dummies for.")
        return df

def split_features_target(df, target_col):
    """
    Separate features (X) and target (y).
    """
    if target_col in df.columns:
        # Drop target column to get features
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        print(f"Separated features (X) and target (y = '{target_col}')")
        return X, y
    else:
        print(f"Error: Target column '{target_col}' not found.")
        return None, None

def split_train_test(X, y, test_size=0.2, random_state=123, stratify=True):
    """
    Split data into training and testing sets.
    """
    # Use stratification if specified
    stratify_param = y if stratify else None
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    print(f"Data split into train and test sets (test_size={test_size}, random_state={random_state}).")
    return xtrain, xtest, ytrain, ytest

def scale_features(xtrain, xtest):
    """
    Scale features using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    # Fit scaler on training data
    scaler.fit(xtrain)
    print("MinMaxScaler fitted on training data.")

    # Transform training and testing data
    xtrain_scaled = scaler.transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    print("Training and testing features scaled.")

    return scaler, xtrain_scaled, xtest_scaled