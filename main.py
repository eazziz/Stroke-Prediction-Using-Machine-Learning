import pandas as pd
from preprocessing import preprocess_data
from model import train_models, predict_stroke
from visualize import plot_stroke_occurrences, plot_model_accuracy

# Load the dataset
df = pd.read_csv('./healthcare-dataset-stroke-data.csv')

# Preprocess the data
X, Y, scaler, df = preprocess_data(df)

# Train models and get predictions
log_reg, knn, Y_pred_log_reg, Y_pred_knn = train_models(X, Y)

# Plot results
plot_stroke_occurrences(df)
plot_model_accuracy(Y, Y_pred_log_reg, Y_pred_knn)

# Ask for user input and predict
predict_stroke(log_reg, knn, scaler)

