import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Earnings Manipulation Bulk Scanner", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model.pkl")
model = joblib.load(MODEL_PATH)

st.title("📊 Earnings Manipulation Bulk Scanner")
st.write("Upload an Excel file with Beneish ratios to detect manipulation risk.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    required_cols = list(model.feature_names_in_)

    if not all(col in df.columns for col in required_cols):
        st.error(f"Excel must contain these columns:\n{required_cols}")
    else:
        X = df[required_cols]
        df["Risk_Prediction"] = model.predict(X)
        df["Risk_Probability"] = model.predict_proba(X)[:,1]

        df["Risk_Prediction"] = df["Risk_Prediction"].map({1:"Likely Manipulator",0:"Likely Non-Manipulator"})

        st.success("Prediction completed!")
        st.dataframe(df)

        st.download_button("Download Results as Excel",
                           data=df.to_excel(index=False),
                           file_name="fraud_results.xlsx")


