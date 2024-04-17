# IMPORT STATEMENTS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import pickle

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

# Use trained model to make predictions on the test set
y_pred = rf.predict(x_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Save trained model
joblib.dump(rf, "model.pkl")

# Calculate performance metrics
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Store metrics in a dictionary
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": cm
}

# Save metrics
with open("metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)
