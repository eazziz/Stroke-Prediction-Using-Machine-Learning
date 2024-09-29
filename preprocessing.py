import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    #Drop missing values and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop('id', axis=1, inplace=True)

    #Encoding categorical variables
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['ever_married'] = le.fit_transform(df['ever_married'])
    df['work_type'] = le.fit_transform(df['work_type'])
    df['Residence_type'] = le.fit_transform(df['Residence_type'])
    df['smoking_status'] = le.fit_transform(df['smoking_status'])

    #Normalization
    scaler = StandardScaler()
    numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    #Address imbalance with SMOTE
    pre_X = df.drop('stroke', axis=1)
    pre_Y = df['stroke']
    smote = SMOTE(random_state=42)
    X, Y = smote.fit_resample(pre_X, pre_Y)

    return X, Y, scaler, df
