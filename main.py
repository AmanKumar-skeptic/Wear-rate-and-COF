import streamlit as st

# Streamlit UI
st.title("Wear Rate and Coefficient of Friction Predictor")

# Input fields
phase = st.selectbox("Select Phase", options=["FCC", "BCC", "FCC+BCC"])  # Replace with actual phase names
hardness = st.number_input("Enter Hardness", min_value=0.0, max_value=100.0, step=0.1)
sliding_distance = st.number_input("Enter Sliding Distance (m)", min_value=0.0, step=0.1)
sliding_velocity = st.number_input("Enter Sliding Velocity (m/s)", min_value=0.0, step=0.1)
load = st.number_input("Enter Load (N)", min_value=0.0, step=0.1)

# Button to simulate prediction
if st.button("Predict"):
    st.success("You are a bit early, We are working on it!")
