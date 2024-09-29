from preprocessing import preprocess_data
from model import train_models, predict_stroke
from visualize import plot_age_group, plot_accuracy

def main():
    # Preprocess the data
    X_train, X_test, Y_train, Y_test, scaler = preprocess_data()

    # Train models and get predictions
    log_reg, knn, Y_pred_log_reg, Y_pred_knn = train_models(X_train, Y_train, X_test)

    # Evaluate models
    evaluate_models(Y_test, Y_pred_log_reg, Y_pred_knn)

    # Plot visualizations
    plot_age_group()
    plot_accuracy()

    # Predict user input
    predict_stroke(log_reg, knn, scaler)

if __name__ == "__main__":
    main()
