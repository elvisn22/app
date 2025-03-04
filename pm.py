import streamlit as st
import joblib
import numpy as np

# Load the trained model
#@st.cache_resource  # Cache the model to avoid reloading it on every interaction
def load_model():
    try:
        return joblib.load('xgb_model_fold_4.joblib')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()

# Title and Description
st.title("Predictive Maintenance Model")
st.write("Provide the input features to predict the target or failure type.")

# Collect user inputs for each feature
air_temp = st.number_input("Air temperature [K]", min_value=200.0, max_value=400.0, value=303.0, step=0.1)
process_temp = st.number_input("Process temperature [K]", min_value=200.0, max_value=400.0, value=311.2, step=0.1)
rotational_speed = st.number_input("Rotational speed [rpm]", min_value=0.0, max_value=5000.0, value=1601.0, step=1.0)
torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=32.9, step=0.1)
tool_wear = st.number_input("Tool wear [min]", min_value=0.0, max_value=300.0, value=69.0, step=1.0)

# Categorical features
type_l = st.selectbox("Type L (1 for Yes, 0 for No)", options=[0, 1], index=1)
type_m = st.selectbox("Type M (1 for Yes, 0 for No)", options=[0, 1], index=1)

# Combine inputs into a feature array
features = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear, type_l, type_m]])

# Define the class labels corresponding to model's output
labels = ['No Failure', 'Power Failure', 'Tool Wear Failure', 'Overstrain Failure', 'Random Failures', 'Heat Dissipation Failure']

# Custom thresholds based on feature combinations
def custom_prediction_logic(features):
    air_temp, process_temp, rotational_speed, torque, tool_wear, type_l, type_m = features[0]

    # Example failure conditions
    if torque > 70:
        return 'Overstrain Failure'
    elif air_temp > 370 and process_temp > 370:
        return 'Heat Dissipation Failure'
    elif tool_wear > 100:
        return 'Tool Wear Failure'
    elif rotational_speed > 3000:
        return 'Power Failure'
    elif type_l == 1 and type_m == 0:
        return 'Random Failures'
    else:
        return 'No Failure'

# Prediction Button
if st.button("Predict"):
    if model is not None:
        # Check the custom logic first
        custom_prediction = custom_prediction_logic(features)
        
        # Show the custom prediction
        st.success(f"The predicted faliure that can happen is: {custom_prediction}")
    else:
        st.error("Model could not be loaded. Check the filename and ensure the model is in the app directory.")
