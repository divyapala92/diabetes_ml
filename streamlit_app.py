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
# Accessing confusion matrix from the performance metrics
st.write("Confusion Matrix:")
st.write(pd.DataFrame({
    "": ["Actual Negative", "Actual Positive"],
    "Predicted Negative": [performance_metrics['TN'], performance_metrics['FN']],
    "Predicted Positive": [performance_metrics['FP'], performance_metrics['TP']]
}))

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
# You need to load the model and make predictions here
# Replace `user_result = rf.predict(user_data_df)` with your prediction code

# Displaying the prediction result
st.subheader('Your Report: ')
# Displaying the prediction result based on your model's prediction
# Replace `output = 'You are Diabetic' if user_result[0] == 1 else 'You are not Diabetic'` with your code

# VISUALIZATIONS
# Your visualization code goes here
