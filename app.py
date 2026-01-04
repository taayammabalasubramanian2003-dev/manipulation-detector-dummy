import streamlit as st
import pandas as pd
import joblib
import os
import io

st.set_page_config(page_title="Earnings Manipulation Bulk Scanner", layout="centered")

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_model.pkl")
model = joblib.load(MODEL_PATH)

st.title("📊 Earnings Manipulation Bulk Scanner")
st.write("Upload an Excel file containing Beneish ratios to detect earnings manipulation risk.")

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

        # Convert dataframe to downloadable Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)

        st.download_button(
            label="Download Results as Excel",
            data=output.getvalue(),
            file_name="fraud_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



