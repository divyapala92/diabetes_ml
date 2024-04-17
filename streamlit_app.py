import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the trained model
loaded_model = joblib.load("trained_model.pkl")

# Load the test data
test_data = pd.read_csv("Clean_BDHS_Diabetic_Data_Jahan_Balanced.csv")
X_test = test_data[['weight', 'height', 'SBP', 'DBP', 'age']]
y_test = test_data['diabetes']

# Make predictions on the test data
y_pred = loaded_model.predict(X_test)

# Ensure both y_test and y_pred are one-dimensional arrays
y_test = y_test.values.ravel()
y_pred = y_pred.ravel()

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display performance metrics
st.write("## Model Performance Metrics")
st.write(f"Accuracy: {accuracy * 100:.2f}%")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# Displaying the confusion matrix
st.write("## Confusion Matrix")
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
