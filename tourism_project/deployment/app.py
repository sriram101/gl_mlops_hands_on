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
st.set_page_config(page_title="Tourism Package Prediction", layout="wide")

# Compact styling
st.markdown("""
<style>
    .block-container {padding-top: 0.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem;}
    h3 {font-size: 1.0rem !important; margin-bottom: 0rem !important; margin-top: 0rem !important;}
    p {font-size: 0.8rem !important; margin-bottom: 0rem !important;}
    .stNumberInput, .stSelectbox {margin-bottom: -15px !important;}
    .stForm {padding: 0.3rem !important;}
    hr {margin-top: 0.3rem !important; margin-bottom: 0.3rem !important;}
    [data-testid="stFormSubmitButton"] {margin-top: -10px !important;}
    .stAlert {padding: 0.5rem !important; font-size: 0.85rem !important;}
    div[data-testid="stExpander"] {font-size: 0.8rem !important;}
</style>
""", unsafe_allow_html=True)

st.subheader("Wellness Tourism Package: Purchase Prediction")

st.divider()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Customer Profile**")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=22000)

    with col2:
        st.markdown("**Travel Details**")
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        num_persons = st.number_input("Persons Visiting", min_value=1, max_value=5, value=3)
        num_children = st.number_input("Children Visiting", min_value=0, max_value=3, value=1)
        num_trips = st.number_input("Number of Trips", min_value=1, max_value=22, value=3)
        passport = st.selectbox("Passport", [0, 1])
        own_car = st.selectbox("Own Car", [0, 1])

    with col3:
        st.markdown("**Sales Interaction**")
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
        duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=5, max_value=130, value=15)
        num_followups = st.number_input("Number of Followups", min_value=1, max_value=6, value=4)
        pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
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

    st.divider()

    with st.expander("Input Summary"):
        st.dataframe(input_data, use_container_width=True)
            
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"Prediction: Purchase probability is {probability[1]:.0%}. This customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.warning(f"Prediction: Purchase probability is {probability[1]:.0%}. This customer is unlikely to purchase the Wellness Tourism Package.")
