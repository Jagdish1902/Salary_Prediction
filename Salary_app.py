import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline (includes preprocessing + XGBoost)

best_xgb = joblib.load('Salary_XGBoost_Pipeline.joblib')

# App title
st.title("💼 Salary Prediction App")

# Load dataset to populate Job Titles dynamically
df = pd.read_csv("Salary_Data.csv")
job_titles = sorted(df["Job Title"].dropna().unique().tolist())

age = st.slider("Age", 18, 65, 30)
education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
job_title = st.selectbox("Job Title", job_titles)
experience = st.slider("Years of Experience", 0, 40, 5)

# --- Create a DataFrame for input ---
input_df = pd.DataFrame({
    "Age": [age],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [experience]
})

# Display profile
st.subheader("📋 Candidate Profile")
st.write(input_df)

# --- Prediction ---
if st.button("Predict Salary"):
    prediction = best_xgb.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: ₹{prediction:,.2f}")


