import streamlit as st
import joblib
import numpy as np

# Load the trained models
with open('rf_crop.joblib', 'rb') as file:
    crop_model = joblib.load(file)

with open('knn_fertilizer.joblib', 'rb') as file:
    fertilizer_model = joblib.load(file)
    
with open('Scaler_crop.joblib', 'rb') as file:
    scaler = joblib.load(file)  # Assuming you saved the scaler during model training

# Label Encoders for the models
crop_label_encoder = {
    0: "Sugarcane", 1: "Wheat", 2: "Cotton", 3: "Jowar", 4: "Rice",
    5: "Maize", 6: "Groundnut", 7: "Grapes", 8: "Tur", 9: "Ginger",
    10: "Turmeric", 11: "Urad", 12: "Gram", 13: "Moong", 14: "Soybean", 15: "Masoor"
}

fertilizer_label_encoder = {
    0: "Urea", 1: "DAP", 2: "MOP", 3: "SSP", 4: "19:19:19 NPK",
    5: "Chilated Micronutrient", 6: "50:26:26 NPK", 7: "Magnesium Sulphate", 
    8: "10:26:26 NPK", 9: "Ferrous Sulphate", 10: "13:32:26 NPK", 
    11: "10:10:10 NPK", 12: "Ammonium Sulphate", 13: "12:32:16 NPK", 
    14: "White Potash", 15: "Hydrated Lime", 16: "20:20:20 NPK", 
    17: "18:46:00 NPK", 18: "Sulphur"
}

# Prediction functions
def predict_crop(Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature):
    # Prepare the input data and scale it
    crop_input = np.array([[Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature]])
    crop_input_scaled = scaler.transform(crop_input)
    
    # Predict the crop
    crop_prediction = crop_model.predict(crop_input_scaled)
    return crop_prediction[0]

def predict_fertilizer(Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Crop):
    # Prepare the input data and scale it
    crop_index = list(crop_label_encoder.keys())[list(crop_label_encoder.values()).index(Crop)]
    fertilizer_input = np.array([[Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, crop_index]])
    fertilizer_input_scaled = scaler.transform(fertilizer_input[:, :-1])
    
    # Add crop index back to the scaled input
    fertilizer_input_scaled = np.hstack([fertilizer_input_scaled, [[crop_index]]])
    
    # Predict the fertilizer
    fertilizer_prediction = fertilizer_model.predict(fertilizer_input_scaled)
    return fertilizer_label_encoder[int(fertilizer_prediction[0])]

# Streamlit App
st.title("Crop and Fertilizer Prediction")

# Handle navigation state
if "page" not in st.session_state:
    st.session_state.page = "welcome"

def go_to_predictions():
    st.session_state.page = "predictions"

if st.session_state.page == "welcome":
    st.header("Welcome to the Crop and Fertilizer Recommendation App")
    st.write("This app helps you predict the best crop to grow and the optimal fertilizer to use based on specific soil conditions and environmental factors.")
    st.button("Go to Prediction Page", on_click=go_to_predictions)

elif st.session_state.page == "predictions":
    # Tabs for Crop Prediction and Fertilizer Prediction
    tab1, tab2 = st.tabs(["Crop Prediction", "Fertilizer Prediction"])

    with tab1:
        st.header("Predict the Best Crop")
        
        # Input fields for crop prediction
        Nitrogen = st.number_input("Nitrogen", min_value=0.0)
        Phosphorus = st.number_input("Phosphorus", min_value=0.0)
        Potassium = st.number_input("Potassium", min_value=0.0)
        pH = st.number_input("pH", min_value=0.0, max_value=14.0)
        Rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
        Temperature = st.number_input("Temperature (°C)", min_value=0.0)
        
        if st.button("Predict Crop"):
            crop = predict_crop(Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature)
            st.success(f"The recommended crop is: {crop}")

    with tab2:
        st.header("Predict the Best Fertilizer")
        
        # Input fields for fertilizer prediction
        Nitrogen = st.number_input("Nitrogen", min_value=0.0, key="fertilizer_nitrogen")
        Phosphorus = st.number_input("Phosphorus", min_value=0.0, key="fertilizer_phosphorus")
        Potassium = st.number_input("Potassium", min_value=0.0, key="fertilizer_potassium")
        pH = st.number_input("pH", min_value=0.0, max_value=14.0, key="fertilizer_ph")
        Rainfall = st.number_input("Rainfall (mm)", min_value=0.0, key="fertilizer_rainfall")
        Temperature = st.number_input("Temperature (°C)", min_value=0.0, key="fertilizer_temperature")
        Crop = st.selectbox("Crop", list(crop_label_encoder.values()))
        
        if st.button("Predict Fertilizer"):
            fertilizer = predict_fertilizer(Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Crop)
            st.success(f"The recommended fertilizer is: {fertilizer}")
