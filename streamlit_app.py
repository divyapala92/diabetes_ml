import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("trained_model.pkl")

# Load the performance metrics directly from the training script output
performance_metrics = {
    "accuracy": 0.85,  # Replace with actual values
    "precision": 0.75,
    "recall": 0.90,
    "f1": 0.82
}

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
# Display confusion matrix and other performance metrics if needed
