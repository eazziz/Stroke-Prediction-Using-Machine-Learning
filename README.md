# Stroke Prediction Project

## Overview

This project aims to predict the likelihood of a stroke based on various health and lifestyle factors. The dataset used includes information about individuals, and the model employs machine learning techniques to provide predictions. 

## Project Flow

1. **Data Loading**:
   - The project begins by importing the necessary libraries for data manipulation, preprocessing, model training, and visualization.
   - The dataset `healthcare-dataset-stroke-data.csv` is loaded into a pandas DataFrame.

2. **Data Preprocessing**:
   - Duplicate rows and missing values are removed to ensure data quality.
   - The 'id' column is dropped as it's not needed for the analysis.
   - Categorical variables such as `gender`, `ever_married`, `work_type`, `Residence_type`, and `smoking_status` are encoded into numerical values using `LabelEncoder`.

3. **Exploratory Data Analysis**:
   - Age groups are created to analyze the occurrences of strokes in different age ranges.
   - A bar graph visualizes the number of strokes across age groups.

4. **Normalization**:
   - Numerical features (`age`, `hypertension`, `heart_disease`, `avg_glucose_level`, and `bmi`) are normalized using `StandardScaler` to ensure all features contribute equally to the model.

5. **Addressing Class Imbalance**:
   - SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset by creating synthetic examples of the minority class.

6. **Train-Test Split**:
   - The dataset is split into training and testing sets to evaluate the model's performance effectively.

7. **Model Training**:
   - Two machine learning models are trained: 
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**

8. **Model Evaluation**:
   - The accuracy, precision, recall, and F2-score for both models are calculated and displayed.
   - A bar graph compares the accuracy of the models visually.

9. **User Input for Prediction**:
   - A function prompts users to input their health and lifestyle details to predict the likelihood of a stroke.
   - The input data is normalized using the previously fitted scaler.

10. **Prediction**:
    - The trained models predict the likelihood of a stroke based on user input, and the results are displayed.

## Running the Project

To run the project, follow these steps:

1. Ensure you have the required libraries installed. You can install them using pip:

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

