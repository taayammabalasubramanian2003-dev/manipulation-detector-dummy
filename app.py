import streamlit as st
import pandas as pd
import joblib
import os

# Safe model loading path (works on Streamlit Cloud)
model_path = os.path.join("model", "final_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Earnings Manipulation Detector", layout="centered")

st.title("🔍 Earnings Manipulation Detector")
st.write("This tool predicts whether a company is likely manipulating earnings using Beneish financial ratios.")

st.subheader("Enter Financial Ratios")

DSRI = st.number_input("DSRI – Days Sales Receivable Index", value=1.0)
GMI = st.number_input("GMI – Gross Margin Index", value=1.0)
AQI = st.number_input("AQI – Asset Quality Index", value=1.0)
SGI = st.number_input("SGI – Sales Growth Index", value=1.0)
DEPI = st.number_input("DEPI – Depreciation Index", value=1.0)
SGAI = st.number_input("SGAI – SG&A Expense Index", value=1.0)
ACCR = st.number_input("ACCR – Total Accruals", value=0.0)
LEVI = st.number_input("LEVI – Leverage Index", value=1.0)

if st.button("Check Manipulation Risk"):
    input_df = pd.DataFrame(
        [[DSRI, GMI, AQI, SGI, DEPI, SGAI, ACCR, LEVI]],
        columns=["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI"]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely Earnings Manipulator\n\nRisk Probability: {probability:.2f}")
    else:
        st.success(f"✅ Likely Non-Manipulator\n\nRisk Probability: {probability:.2f}")

st.markdown("---")
st.caption("ML model trained using Beneish ratios on Indian firm financial data.")

