from data_preprocessing import load_and_preprocess_data
from model import train_models, predict_stroke
from visualization import plot_age_groups, plot_accuracy_comparison

# Load and preprocess data
df, scaler = load_and_preprocess_data()

# Train models and get predictions
log_reg, knn = train_models(df)

# Visualize age group data
plot_age_groups(df)

# Visualize accuracy comparison
plot_accuracy_comparison(log_reg, knn)

# Ask for user input and predict stroke
predict_stroke(log_reg, knn, scaler)
