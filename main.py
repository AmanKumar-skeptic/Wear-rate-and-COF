import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from xgboost import XGBRegressor

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('D:/Compressive_Sterngth/Mega Dataset_Wear Dataset - Compilation.csv')
    return data

# Train model
@st.cache_data
def train_model(data):
    encoder = TargetEncoder(smoothing=5)
    data['encoded_combination'] = encoder.fit_transform(data['Combination'], data['Wear Rate'])
    
    X = data.drop(columns=['Combination', 'Phase', 'Wear Rate'])
    y = data['Wear Rate']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.005, max_depth=3, random_state=42)
    model.fit(X, y)
    
    return model, encoder

def main():
    st.title("Wear Rate Prediction App")
    
    # Load data
    data = load_data()
    
    # Train model
    model, encoder = train_model(data)
    
    # Get unique combinations
    combinations = sorted(data['Combination'].unique())
    
    # Create input form
    st.header("Input Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        combination = st.selectbox("Select Combination", combinations)
        load = st.slider("Load", min_value=10, max_value=30, step=10)
    
    with col2:
        sliding_distance = st.slider("Sliding Distance", min_value=500, max_value=1500, step=500)
    
    with col3:
        sliding_speed = st.slider("Sliding Speed", min_value=1, max_value=3, step=1)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Combination': [combination],
        'Load': [load],
        'Sliding Distance': [sliding_distance],
        'Sliding Speed': [sliding_speed]
    })
    
    # Encode combination
    input_data['encoded_combination'] = encoder.transform(input_data['Combination'])
    
    # Prepare features for prediction
    X_pred = input_data.drop(columns=['Combination'])
    
    # Make prediction
    if st.button("Predict Wear Rate"):
        prediction = model.predict(X_pred)[0]
        
        # Get actual wear rate for comparison
        actual_data = data[
            (data['Combination'] == combination) &
            (data['Load'] == load) &
            (data['Sliding Distance'] == sliding_distance) &
            (data['Sliding Speed'] == sliding_speed)
        ]
        
        if not actual_data.empty:
            actual_wear_rate = actual_data['Wear Rate'].mean()
            error = abs(prediction - actual_wear_rate) / actual_wear_rate * 100
            
            st.subheader("Results")
            st.write(f"Predicted Wear Rate: {prediction:.3f}")
            st.write(f"Actual Wear Rate: {actual_wear_rate:.3f}")
            st.write(f"Error: {error:.2f}%")
            
            if error <= 5:
                st.success("Prediction is within ±5% of actual value!")
            else:
                st.warning("Prediction is outside ±5% range of actual value.")
        else:
            st.write(f"Predicted Wear Rate: {prediction:.3f}")
            st.info("No actual data available for this combination of parameters.")

if __name__ == "__main__":
    main()
