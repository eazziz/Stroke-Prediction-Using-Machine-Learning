import matplotlib.pyplot as plt
import seaborn as sns

def plot_stroke_occurrences(df):
    # Create age groups
    stroke_data = df[df['stroke'] == 1]
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]
    labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
    stroke_data['age_group'] = pd.cut(stroke_data['age'], bins=bins, labels=labels, right=False)

    # Count occurrences of strokes in each age group
    age_group_counts = stroke_data['age_group'].value_counts().sort_index()

    # Create a bar graph
    plt.figure(figsize=(12, 6))
    plt.bar(age_group_counts.index, age_group_counts.values, color='blue', alpha=0.7)
    plt.title('Occurrences of Stroke by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Strokes')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

def plot_model_accuracy(Y, Y_pred_log_reg, Y_pred_knn):
    from sklearn.metrics import accuracy_score

    accuracy_log = accuracy_score(Y, Y_pred_log_reg)
    accuracy_knn = accuracy_score(Y, Y_pred_knn)

    models = ['Logistic Regression', 'KNN']
    accuracies = [accuracy_log, accuracy_knn]

    fig, ax = plt.subplots(figsize=(8, 6))
    bar_colors = ['#1f77b4', '#ff7f0e']
    ax.bar(models, accuracies, color=bar_colors, width=0.4)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison of Models', fontsize=14)

    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.01, f'{acc:.2f}', ha='center', fontsize=12)

    plt.show()
