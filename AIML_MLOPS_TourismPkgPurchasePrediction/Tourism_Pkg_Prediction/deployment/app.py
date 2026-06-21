import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Sasimscct/Tourism-Package-Purchase-Prediction", filename="best_tourism_pkg_predict_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of purchasing the Wellness Tourism Package based on its operational parameters.
Please enter the customer data below to get a prediction.
""")

# User input
type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Divorced", "Married"]
)

designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

age = st.number_input(
    "Age",
    min_value=18.0,
    max_value=100.0,
    value=35.0,
    step=1.0
)

city_tier = st.number_input(
    "City Tier",
    min_value=1,
    max_value=3,
    value=2,
    step=1
)

duration_of_pitch = st.number_input(
    "Duration Of Pitch",
    min_value=0.0,
    value=15.0,
    step=1.0
)

number_of_person_visiting = st.number_input(
    "Number Of Person Visiting",
    min_value=1,
    value=2,
    step=1
)

number_of_followups = st.number_input(
    "Number Of Followups",
    min_value=0.0,
    value=2.0,
    step=1.0
)

preferred_property_star = st.number_input(
    "Preferred Property Star",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=1.0
)

number_of_trips = st.number_input(
    "Number Of Trips",
    min_value=0.0,
    value=2.0,
    step=1.0
)

passport = st.number_input(
    "Passport (0 = No, 1 = Yes)",
    min_value=0,
    max_value=1,
    value=0,
    step=1
)

pitch_satisfaction_score = st.number_input(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

own_car = st.number_input(
    "Own Car (0 = No, 1 = Yes)",
    min_value=0,
    max_value=1,
    value=0,
    step=1
)

number_of_children_visiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0.0,
    value=0.0,
    step=1.0
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=30000.0,
    step=1000.0
)

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Wellness Tourism Package"):
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase Wellness Tourism Package" if prediction == 1 else "Customer will not purchase Wellness Tourism Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
