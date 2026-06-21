import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="neelamsh/Tourist-Prediction", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Visit With Us Tour Pack Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package or not.
Please enter the customer details below to get a prediction.
""")

# User input
st.subheader("Customer Information")

Age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free lancer"])
MonthlyIncome = st.number_input("Monthly Income ($)", min_value=0, max_value=1000000, value=50000, step=1000)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

st.subheader("Location & Family Details")

CityTier = st.selectbox("City Tier", [1, 2, 3])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, step=1)
OwnCar = st.toggle("Owns a Car")

st.subheader("Travel History")

NumberOfTrips = st.number_input("Number of Trips (Annual)", min_value=0, max_value=50, value=3, step=1)
Passport = st.toggle("Has Passport")
PreferredPropertyStar = st.slider("Preferred Hotel Star Rating", min_value=3, max_value=5, value=3, step=1)

st.subheader("Sales Interaction Details")

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=150, value=15, step=1)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=20, value=3, step=1)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': int(Passport), # True/False into int
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': int(OwnCar), # True/False into int
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Customer Decision"):
    prediction = model.predict(input_data)[0]
    result = "Tour Pack Taken" if prediction == 1 else "Tour Pack Not Taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
