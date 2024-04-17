import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the trained model
loaded_model = joblib.load("trained_model.pkl")

# Streamlit UI
st.title('Diabetes Checkup')

# Load performance metrics directly from the trained model
y_test = pd.read_csv("Clean_BDHS_Diabetic_Data_Jahan_Balanced.csv")['diabetes']
x_test = pd.read_csv("Clean_BDHS_Diabetic_Data_Jahan_Balanced.csv")[['weight', 'height', 'SBP', 'DBP', 'age']]

# Calculate performance metrics
y_pred = loaded_model.predict(x_test)
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

# Streamlit UI continued...
