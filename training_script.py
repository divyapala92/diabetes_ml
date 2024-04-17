import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

# Read the dataset
df = pd.read_csv("Clean_BDHS_Diabetic_Data_Jahan_Balanced.csv")

# Label Encoding
le = LabelEncoder()
df['diabetes'] = le.fit_transform(df['diabetes'])

# Define features and target variable based on specified important features
X = df[['weight', 'height', 'SBP', 'DBP', 'age']]
y = df['diabetes']

# Splitting the dataset into training and test sets using train_test_split function from scikit learn
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialise and train a random forest classifier using training data
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Save the trained model to a file
joblib.dump(rf, "trained_model.pkl")

# Save the label encoder to a file
joblib.dump(le, "label_encoder.pkl")

# Save performance metrics to a file
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

performance_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}

with open("performance_metrics.txt", "w") as f:
    for metric, value in performance_metrics.items():
        f.write(f"{metric}: {value}\n")
