import streamlit as st
import pandas as pd
import joblib
import os
import io

st.set_page_config(page_title="Earnings Manipulation Detector", layout="centered")

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model.pkl")
model = joblib.load(MODEL_PATH)

st.title("🔍 Earnings Manipulation Detector")
st.write("Detect earnings manipulation risk using Beneish financial ratios")

mode = st.radio("Select Mode", ["Bulk Excel Scanner", "Manual Entry Checker"])

# ================= BULK MODE =================
if mode == "Bulk Excel Scanner":
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        required_cols = list(model.feature_names_in_)

        if not all(col in df.columns for col in required_cols):
            st.error(f"Excel must contain these columns:\n{required_cols}")
        else:
            X = df[required_cols]
            df["Risk_Prediction"] = model.predict(X)
            df["Risk_Probability"] = model.predict_proba(X)[:, 1]
            df["Risk_Prediction"] = df["Risk_Prediction"].map({1: "Likely Manipulator", 0: "Likely Non-Manipulator"})

            st.success("Prediction completed!")
            st.dataframe(df)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)

            st.download_button("Download Results",
                               data=output.getvalue(),
                               file_name="fraud_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ================= MANUAL MODE =================
if mode == "Manual Entry Checker":
    st.subheader("Enter Beneish Ratios")

    DSRI = st.number_input("DSRI", value=1.0)
    GMI = st.number_input("GMI", value=1.0)
    AQI = st.number_input("AQI", value=1.0)
    SGI = st.number_input("SGI", value=1.0)
    DEPI = st.number_input("DEPI", value=1.0)
    SGAI = st.number_input("SGAI", value=1.0)
    ACCR = st.number_input("ACCR", value=0.0)
    LEVI = st.number_input("LEVI", value=1.0)

    if st.button("Check Manipulation Risk"):
        feature_order = list(model.feature_names_in_)
        user_vals = [DSRI, GMI, AQI, SGI, DEPI, SGAI, ACCR, LEVI]

        # Auto-fill any extra training columns (like Company_ID)
        while len(user_vals) < len(feature_order):
            user_vals.insert(0, 0)

        input_df = pd.DataFrame([user_vals], columns=feature_order)

        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠ Likely Manipulator (Risk Probability: {prob:.2f})")
        else:
            st.success(f"✅ Likely Non-Manipulator (Risk Probability: {prob:.2f})")






