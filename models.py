from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

def train_models(X, Y):
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)
    Y_pred_log_reg = log_reg.predict(X_test)

    # KNN Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)
    Y_pred_knn = knn.predict(X_test)

    return log_reg, knn, Y_pred_log_reg, Y_pred_knn

def predict_stroke(log_reg, knn, scaler):
    user_data = get_user_input()
    new_user_data = normalize_user_data(user_data, scaler)

    # Use trained model to predict
    prediction_log = log_reg.predict(new_user_data)
    prediction_knn = knn.predict(new_user_data)

    # Display results
    print("\nPrediction Results:")
    print("Logistic Regression Model predicts a stroke." if prediction_log == 1 else "Logistic Regression Model predicts no stroke.")
    print("KNN Model predicts a stroke." if prediction_knn == 1 else "KNN Model predicts no stroke.")

def get_user_input():
    print("Enter the following details to predict the likelihood of stroke:")
    gender = int(input("Gender (0 = Male, 1 = Female): "))
    age = float(input("Age: "))
    hypertension = int(input("Hypertension (0 = No, 1 = Yes): "))
    heart_disease = int(input("Heart Disease (0 = No, 1 = Yes): "))
    ever_married = int(input("Ever Married (0 = No, 1 = Yes): "))
    work_type = int(input("Work Type (0 = Private, 1 = Self-employed, 2 = Children, 3 = Govt_job, 4 = Never worked): "))
    residence_type = int(input("Residence Type (0 = Urban, 1 = Rural): "))
    avg_glucose_level = float(input("Average Glucose Level: "))
    bmi = float(input("BMI: "))
    smoking_status = int(input("Smoking Status (0 = Never smoked, 1 = Unknown, 2 = Formerly smoked, 3 = Smokes): "))

    # Create a DataFrame from the input data
    user_data = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]],
                             columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

    return user_data

def normalize_user_data(user_data, scaler):
    numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    user_data[numerical_features] = scaler.transform(user_data[numerical_features])
    return user_data
