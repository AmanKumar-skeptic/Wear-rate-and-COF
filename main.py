import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import pickle

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('Appendix1_Mega Dataset_Wear Dataset.csv')
    return data

# Load the pre-trained model and create encoder
@st.cache_resource
def load_model_and_encoder():
    with open('rf_hea_wr.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Create and fit the target encoder
    data = load_data()
    encoder = TargetEncoder(smoothing=5)
    encoder.fit(data[['Combination']], data['Wear Rate'])
    
    return model, encoder

def main():
    st.title("Wear Rate Prediction App")
    
    # Load data
    data = load_data()
    
    # Load pre-trained model and encoder
    model, encoder = load_model_and_encoder()
    
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
        phase = st.selectbox("Phase", ["FCC", "BCC", "BCC_FCC", "MIP"])
    
    # Create input dataframe with exact feature names and order expected by the model
    input_data = pd.DataFrame({
        'Load': [load],
        'Sliding Distance': [sliding_distance],
        'Sliding Speed': [sliding_speed],
        'Phase_BCC': [1 if phase == "BCC" else 0],
        'Phase_BCC_FCC': [1 if phase == "BCC_FCC" else 0],
        'Phase_FCC': [1 if phase == "FCC" else 0],
        'Phase_MIP': [1 if phase == "MIP" else 0]
    })
    
    # Add encoded combination
    combination_df = pd.DataFrame({'Combination': [combination]})
    input_data['encoded_combination'] = encoder.transform(combination_df)
    
    # Reorder columns to match the training data order
    input_data = input_data[[
        'Load',
        'Sliding Distance',
        'Sliding Speed',
        'encoded_combination',
        'Phase_BCC',
        'Phase_BCC_FCC',
        'Phase_FCC',
        'Phase_MIP'
    ]]
    
    # Make prediction
    if st.button("Predict Wear Rate"):
        prediction = model.predict(input_data)[0]
        
        # Get actual wear rate for comparison
        actual_data = data[
            (data['Combination'] == combination) &
            (data['Load'] == load) &
            (data['Sliding Distance'] == sliding_distance) &
            (data['Sliding Speed'] == sliding_speed) &
            (data['Phase'] == phase)
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
