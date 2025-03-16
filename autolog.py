import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Pranjalk23', repo_name='ML-MLOops-using-MLFlow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Pranjalk23/ML-MLOops-using-MLFlow.mlflow")

# Load Wine dataset
wine = load_wine()
x = wine.data
y = wine.target

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5

# Mention your experiment below
mlflow.autolog()
mlflow.set_experiment('ML-MLOops-using-MLFlow')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Pranjal', "Project": "Wine Classification"})

    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print(accuracy) 