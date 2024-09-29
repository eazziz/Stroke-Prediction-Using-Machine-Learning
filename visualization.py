import matplotlib.pyplot as plt
import seaborn as sns

def plot_age_groups(df):
    stroke_data = df[df['stroke'] == 1]
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]
    labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
    stroke_data['age_group'] = pd.cut(stroke_data['age'], bins=bins, labels=labels, right=False)

    age_group_counts = stroke_data['age_group'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.bar(age_group_counts.index, age_group_counts.values, color='blue', alpha=0.7)
    plt.title('Occurrences of Stroke by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Strokes')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

def plot_accuracy_comparison(log_reg, knn):
    # Assuming you have accuracy scores from these models
    accuracy_log = accuracy_score(...)
    accuracy_knn = accuracy_score(...)

    models = ['Logistic Regression', 'KNN']
    accuracies = [accuracy_log, accuracy_knn]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'], width=0.4)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison of Models')

    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.01, f'{acc:.2f}', ha='center')

    plt.show()
