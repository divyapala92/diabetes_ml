import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Import performance metrics from the training script
from training_script import performance_metrics

# Streamlit UI
st.title('Diabetes Checkup')

# Display model performance metrics
st.write("## Model Performance Metrics")
st.write(f"Accuracy: {performance_metrics['accuracy'] * 100:.2f}%")
st.write(f"Precision: {performance_metrics['precision']:.2f}")
st.write(f"Recall: {performance_metrics['recall']:.2f}")
st.write(f"F1 Score: {performance_metrics['f1']:.2f}")

# Displaying the confusion matrix and performance metrics
st.write("## Confusion Matrix and Performance Metrics")

# Load the trained model
rf = joblib.load("trained_model.pkl")

# Read the dataset for reference
df = pd.read_csv("Clean_BDHS_Diabetic_Data_Jahan_Balanced.csv")

st.sidebar.header('Patient Data Input')

# Creating sliders for input features
user_data = {}
for feature in ['weight', 'height', 'SBP', 'DBP']:
    user_data[feature] = st.sidebar.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

# Modify the age slider to cover a wider age range
user_data['age'] = st.sidebar.slider("Age", 35, 49, 40)  # Adjust the age range as needed

user_data_df = pd.DataFrame([user_data])

# Displaying user input features
st.subheader('Patient Data')
st.write(user_data_df)

# Use the trained model for prediction on user input
user_result = rf.predict(user_data_df)

# Calculate confusion matrix based on user's prediction and true values
# This will give you TN, FP, FN, and TP
y_true = [0]  # Assuming the true label for the user's input is unknown
conf_matrix = confusion_matrix(y_true, user_result)

# Display the confusion matrix
st.write("Confusion Matrix:")
st.write(pd.DataFrame({
    "": ["Actual Negative", "Actual Positive"],
    "Predicted Negative": [conf_matrix[0][0], conf_matrix[1][0]],
    "Predicted Positive": [conf_matrix[0][1], conf_matrix[1][1]]
}))

# Displaying the prediction result
st.subheader('Your Report: ')
output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'
st.title(output)

# Define color based on prediction result
color = 'red' if user_result[0] == 1 else 'blue'

# VISUALIZATIONS
# Your visualization code goes here
