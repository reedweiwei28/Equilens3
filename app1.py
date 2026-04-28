
import streamlit as st
import pandas as pd
from logic import prepare_data, compute_fairness, threshold_analysis

st.title("AI Fairness Audit Tool")

uploaded = st.file_uploader("Upload dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.write("Preview", df.head())

    sensitive = st.selectbox("Select Sensitive Attribute", df.columns)

    if st.button("Run Audit"):
        df = prepare_data(df)

        st.subheader("Baseline Fairness")
        st.write(compute_fairness(df, "approved", sensitive))

        st.subheader("Mitigated Fairness")
        st.write(compute_fairness(df, "approved_mitigated", sensitive))

        st.subheader("Threshold Trade-off")
        results = threshold_analysis(df, "mitigated_score", sensitive)

        for r in results:
            st.write(f"Threshold: {r['threshold']}")
            st.write("Approval Rate:", r["approval_rate"])
            st.write(r["fairness"])
