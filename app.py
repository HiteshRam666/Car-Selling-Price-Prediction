import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the saved model, encoders, and scaler
model = tf.keras.models.load_model('car_price_model.h5')
label_encoders = joblib.load('label_encoders.joblib')
scaler = joblib.load('scaler.joblib')

# Sidebar for project description
st.sidebar.title("About the Project")
st.sidebar.info(
    """
    This project is a **Car Selling Price Predictor** built using an Artificial Neural Network (ANN).
    It predicts the selling price of a car based on several features like:
    
    - Brand
    - Fuel Type
    - Seller Type
    - Transmission Type
    - Owner Type
    - Year of Purchase
    - Kilometers Driven
    
    The model has been trained using historical car data to provide accurate price estimates.
    Use the inputs on the main screen to get predictions.
    """
)

# Main title
st.title("Car Selling Price Predictor🚘🚗")

# Input features
brand = st.selectbox("Select Brand", label_encoders['Brand'].classes_)
fuel = st.selectbox("Select Fuel Type", label_encoders['Fuel'].classes_)
seller_type = st.selectbox("Select Seller Type", label_encoders['Seller_Type'].classes_)
transmission = st.selectbox("Select Transmission", label_encoders['Transmission'].classes_)
owner = st.selectbox("Select Owner Type", label_encoders['Owner'].classes_)
year = st.number_input("Year of Purchase", min_value=1992, max_value=2020, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=500)

# Preprocess the input data
def preprocess_input(brand, fuel, seller_type, transmission, owner, year, km_driven):
    # Encode categorical features using LabelEncoders
    brand_encoded = label_encoders['Brand'].transform([brand])[0]
    fuel_encoded = label_encoders['Fuel'].transform([fuel])[0]
    seller_type_encoded = label_encoders['Seller_Type'].transform([seller_type])[0]
    transmission_encoded = label_encoders['Transmission'].transform([transmission])[0]
    owner_encoded = label_encoders['Owner'].transform([owner])[0]
    
    # Scale the numerical features (year, km_driven)
    features = np.array([[year, km_driven]])
    scaled_features = scaler.transform(features)
    
    # Combine all features into a single array
    input_data = np.array([brand_encoded, fuel_encoded, seller_type_encoded, transmission_encoded, owner_encoded])
    input_data = np.concatenate((input_data, scaled_features[0]))
    
    return input_data.reshape(1, -1)

# Predict button logic
if st.button("Predict Selling Price"):
    input_data = preprocess_input(brand, fuel, seller_type, transmission, owner, year, km_driven)
    
    # Make the prediction
    prediction = model.predict(input_data)
    predicted_price = float(prediction[0])
    
    # Display the prediction
    st.success(f"The predicted selling price is: ₹ {predicted_price:.2f}")
    
    # 1. Scatter Plot: Kilometers Driven vs Predicted Price
    # fig1, ax1 = plt.subplots()
    # ax1.scatter(km_driven, predicted_price, color='blue', label="Predicted Price")
    # ax1.set_xlabel('Kilometers Driven')
    # ax1.set_ylabel('Predicted Selling Price (₹)')
    # ax1.set_title("Kilometers Driven vs Predicted Selling Price")
    # ax1.legend()
    # st.pyplot(fig1)
    
    # 2. Box Plot: Predicted Price by Car Brand
    # fig2, ax2 = plt.subplots()
    # brands = label_encoders['Brand'].classes_
    # price_by_brand = np.random.normal(loc=predicted_price, scale=50000, size=len(brands))  # Simulated price for each brand
    # sns.boxplot(x=brands, y=price_by_brand, ax=ax2)
    # ax2.set_title("Predicted Price by Car Brand")
    # ax2.set_ylabel("Predicted Selling Price (₹)")
    # st.pyplot(fig2)

   # 3. Distribution Plot: Predicted Selling Price Distribution
    fig3, ax3 = plt.subplots()
    prices = np.random.normal(loc=predicted_price, scale=50000, size=100)  # Simulate distribution around predicted price
    sns.histplot(prices, kde=True, ax=ax3)
    ax3.set_title("Distribution of Predicted Prices")
    st.pyplot(fig3)
    
    # # 4. Bar Plot: Average Predicted Price by Fuel Type
    # fig4, ax4 = plt.subplots()
    # fuel_types = label_encoders['Fuel'].classes_
    # avg_price_by_fuel = np.random.normal(loc=predicted_price, scale=50000, size=len(fuel_types))  # Simulated average prices
    # sns.barplot(x=fuel_types, y=avg_price_by_fuel, ax=ax4)
    # ax4.set_title("Average Predicted Price by Fuel Type")
    # ax4.set_ylabel("Predicted Selling Price (₹)")
    # st.pyplot(fig4)