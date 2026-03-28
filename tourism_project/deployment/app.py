# Streamlit frontend for Wellness Tourism Package purchase prediction

import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load the trained model from the Hugging Face model hub
model_path = hf_hub_download(
    repo_id="sriram-acad/tourism-model",
    filename="best_tourism_model.joblib"
)
model = joblib.load(model_path)

# Page configuration
st.title("Wellness Tourism Package: Purchase Prediction")
st.write("Enter customer details to predict the likelihood of purchasing the package.")

# Input fields arranged in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=5, max_value=130, value=15)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=3)
    num_followups = st.number_input("Number of Followups", min_value=1, max_value=6, value=4)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=22000)

with col2:
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Number of Trips", min_value=1, max_value=22, value=3)
    passport = st.selectbox("Passport", [0, 1])
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car = st.selectbox("Own Car", [0, 1])
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Assemble inputs into a dataframe matching the training feature set
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# Display the input data
st.subheader("Customer Input Summary")
st.write(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"Likely to purchase the package (probability: {probability[1]:.2f})")
    else:
        st.warning(f"Unlikely to purchase the package (purchase probability: {probability[1]:.2f})")
