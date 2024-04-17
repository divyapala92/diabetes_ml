# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import subprocess
import json

# Load the trained model
model = joblib.load("trained_model.pkl")

# Execute the training script to get performance metrics
training_script_output = subprocess.run(
    ["python", "training_script.py"],
    capture_output=True,
    text=True
).stdout

# Parse the output to get performance metrics
performance_metrics = json.loads(training_script_output)

# Streamlit UI
st.title('Diabetes Checkup')

# Display model performance metrics
st.write("## Model Performance Metrics")
st.write("Training Metrics:", performance_metrics)  # Display the training metrics directly

# Sidebar for user input
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
user_result = model.predict(user_data_df)

# Displaying the prediction result
st.subheader('Your Report: ')
output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'
st.title(output)

# Define color based on prediction result
color = 'red' if user_result[0] == 1 else 'blue'

# Remaining Streamlit app code for data visualization
# ...
