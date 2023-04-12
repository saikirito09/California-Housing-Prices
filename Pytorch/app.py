import streamlit as st
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from training import HousingNN

# Load the dataset
data = pd.read_csv("housing.csv")

# Load the saved model
model = HousingNN(data.shape[1])
model.load_state_dict(torch.load("housing_model.pt"))

# Input form
st.set_page_config(page_title="Housing Price Predictor", page_icon=":house:")
st.title("Housing Price Predictor")
st.subheader("Enter the Input Features")
longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

# Preprocess the input
encoder = OneHotEncoder(sparse=False)
proximity_encoded = encoder.fit_transform(data['ocean_proximity'].values.reshape(-1, 1))
input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})
input_encoded = pd.concat([
    input_data.drop(columns=["ocean_proximity"]),
    pd.DataFrame(proximity_encoded, columns=encoder.get_feature_names_out(['OP']))
], axis=1)
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_encoded)

# Predict the housing price
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).view(1, -1)
model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)
    predicted_price = output_tensor.item()

# Show the predicted housing price
st.subheader("Predicted Housing Price")
st.write(f"${predicted_price:,.2f}")
