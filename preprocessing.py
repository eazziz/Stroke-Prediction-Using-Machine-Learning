import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    df = pd.read_csv('./healthcare-dataset-stroke-data.csv')
    # Data cleaning steps
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop('id', axis=1, inplace=True)

    # Encoding categorical variables
    le = LabelEncoder()
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Normalization
    scaler = StandardScaler()
    numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Handle imbalance with SMOTE
    pre_X = df.drop('stroke', axis=1)
    pre_Y = df['stroke']
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(pre_X, pre_Y)

    return df, scaler
