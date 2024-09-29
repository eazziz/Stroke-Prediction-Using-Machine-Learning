# Stroke Prediction Project

## Overview

Hey there! Welcome to the Stroke Prediction Project. This project aims to predict the likelihood of a stroke based on various health and lifestyle factors. We’re using a dataset packed with info about individuals, and we’ll employ some cool machine learning techniques to make our predictions.

## Project Flow

1. **Data Loading**:
   - We kick things off by importing the libraries we need for data manipulation, preprocessing, model training, and visualization.
   - The dataset `healthcare-dataset-stroke-data.csv` is loaded into a pandas DataFrame.

2. **Data Preprocessing**:
   - We clean up the data by removing duplicates and missing values to ensure everything's in tip-top shape.
   - The 'id' column is dropped since it’s not really useful for our analysis.
   - Categorical variables like `gender`, `ever_married`, `work_type`, `Residence_type`, and `smoking_status` are turned into numerical values using `LabelEncoder`.

3. **Exploratory Data Analysis**:
   - We create age groups to check out how strokes vary across different age ranges.
   - A bar graph visualizes the number of strokes by age group.  
   <img width="997" alt="age_graph" src="https://github.com/user-attachments/assets/680d1914-9ef1-4c6c-b275-44bb1df19514">

4. **Normalization**:
   - We normalize numerical features (`age`, `hypertension`, `heart_disease`, `avg_glucose_level`, and `bmi`) using `StandardScaler`, ensuring that all features are on the same scale.

5. **Addressing Class Imbalance**:
   - To tackle class imbalance, we use SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of the minority class.

6. **Train-Test Split**:
   - The dataset is split into training and testing sets, which is super important for evaluating our models.

7. **Model Training**:
   - We train two machine learning models:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**

8. **Model Evaluation**:
   - We calculate and display metrics like accuracy, precision, recall, and F2-score for both models.
   - A bar graph compares the accuracy of the models.  
   ![Model Accuracy Comparison](path/to/your/model_accuracy_graph.png)

9. **User Input for Prediction**:
   - There’s a function that asks users to input their health and lifestyle details to predict their likelihood of having a stroke.
   - We normalize the input data using the fitted scaler to keep everything consistent.

10. **Prediction**:
    - Finally, we use the trained models to predict the likelihood of a stroke based on user input and show the results.

## Running the Project

Wanna give it a shot? Here’s how:

1. Make sure you have the required libraries installed. You can get them with pip:

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
