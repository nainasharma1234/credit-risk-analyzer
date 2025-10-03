import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/credit_pipeline.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

st.title("💳 Credit Risk Analyzer — One-tap Analysis")
st.write("Upload CSV of applicants or paste a single row. The model outputs default probability and a credit-score map (300–850).")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("🔍 One-tap Analyze"):
        with st.spinner("Running model..."):
            probs = model.predict_proba(df)[:,1]
            scores = 300 + (1 - probs) * 550
            df_result = df.copy()
            df_result['Default_Probability'] = np.round(probs, 4)
            df_result['Credit_Score'] = np.round(scores, 2)

            st.subheader("Results")
            st.dataframe(df_result[['Default_Probability','Credit_Score']].head())

            st.subheader("Credit Score Distribution")
            st.bar_chart(df_result['Credit_Score'])

            st.subheader("Explainability")
            try:
                clf_wrapper = model.named_steps['clf']
                preproc = model.named_steps['pre']
                X_trans = preproc.transform(df)

                explain_mode = st.radio(
                    "Choose explainability mode:",
                    ["Fast (LogisticRegression only)", "Faithful (Calibrated model)"],
                    index=1,  
                    help="Fast = ignores calibration but quick; Faithful = matches your probability outputs but slower."
                )

                if explain_mode == "Fast (LogisticRegression only)":
                    if hasattr(clf_wrapper, "base_estimator"):
                        base_model = clf_wrapper.base_estimator
                    else:
                        base_model = clf_wrapper

                    st.info("Explaining LogisticRegression (ignoring calibration).")
                    explainer = shap.Explainer(base_model, X_trans)
                    shap_vals = explainer(X_trans).values

                else: 
                    st.info("Explaining calibrated model probabilities (slower, but faithful).")

                    def predict_fn(X):
                        return clf_wrapper.predict_proba(X)[:, 1]

                    explainer = shap.Explainer(predict_fn, X_trans)
                    shap_vals = explainer(X_trans).values

                fig, ax = plt.subplots(figsize=(8, 4))
                shap.summary_plot(shap_vals, X_trans, feature_names=feature_names, show=False)
                st.pyplot(fig)

                st.markdown("**Top contributing features for first record**")
                sample_idx = 0
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_vals[sample_idx] if shap_vals.ndim > 1 else shap_vals
                })
                shap_df['abs'] = np.abs(shap_df['shap_value'])
                shap_df = shap_df.sort_values('abs', ascending=False).head(10)
                st.table(shap_df[['feature', 'shap_value']].set_index('feature'))

            except Exception as e:
                st.error("Explainability failed: " + str(e))



        st.success("Analysis complete.")
else:
    st.subheader("Input Data")
    st.markdown("Upload a CSV file with applicant data. The dataset should include features like `loan_amnt`, `annual_inc`, `age`, `term`, `purpose`, `home_ownership`, `emp_length`, `delinq_2yrs`, `revolving_util`, and `open_acc`.")
    st.markdown("Alternatively, paste a single row of data in CSV format below:")
    sample_input = """loan_amnt,annual_inc,age,term,purpose,home_ownership,emp_length,delinq_2yrs,revolving_util,open_acc"""
   
st.markdown("---")
st.markdown("If you don't have a dataset to test, run `python generate_data.py` then `python train_model.py` to produce sample data & a trained model.")
