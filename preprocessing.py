import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data():
    df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
    # Data cleaning steps here (drop duplicates, handle missing values)
    # Encoding and normalization steps here

    pre_X = df.drop('stroke', axis=1)
    pre_Y = df['stroke']

    # Addressing imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(pre_X, pre_Y)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test, scaler  # Return the scaler if you need it later
