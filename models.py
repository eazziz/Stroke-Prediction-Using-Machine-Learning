from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_models(df):
    # Split the data into training and test sets
    X = df.drop('stroke', axis=1)
    Y = df['stroke']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)

    return log_reg, knn

def predict_stroke(log_reg, knn, scaler):
    user_data = get_user_input()
    new_user_data = normalize_user_data(user_data, scaler)

    # Use trained model to predict
    prediction_log = log_reg.predict(new_user_data)
    prediction_knn = knn.predict(new_user_data)

    # Display results
    print("\nPrediction Results:")
    display_prediction(prediction_log, "Logistic Regression")
    display_prediction(prediction_knn, "KNN")
    
def display_prediction(prediction, model_name):
    if prediction == 1:
        print(f"{model_name} Model predicts a stroke.")
    else:
        print(f"{model_name} Model predicts no stroke.")
