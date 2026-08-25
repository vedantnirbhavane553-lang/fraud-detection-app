import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model safely
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

model = load_model("fraud_detection_random_forest.pkl")

# Helper function to preprocess data
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    
    # Ensure step exists
    if "step" not in processed.columns:
        processed["step"] = 1
        
    # One-hot encode transaction type
    for col_type in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        processed[f"type_{col_type}"] = (processed["type"] == col_type).astype(int)
        
    required_cols = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "type_CASH_OUT",
        "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
    ]
    return processed[required_cols]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("FraudShield AI")
    st.caption("Real-Time Financial Risk Assessment Engine")
    st.markdown("---")
    
    st.subheader("Model Status")
    if model is not None:
        st.success("🟢 Model Loaded: Random Forest")
    else:
        st.error("🔴 Model Missing: Check `.pkl` path")

    st.markdown("---")
    st.info("💡 **Tip:** High-risk indicators often include total drainage of sender balance combined with `TRANSFER` or `CASH_OUT` actions.")

# --- MAIN CONTENT ---
st.title("Transaction Intelligence & Fraud Detection")

tab1, tab2 = st.tabs(["🔍 Real-Time Transaction Check", "📁 Batch Analysis (CSV)"])

# TAB 1: Single Prediction
with tab1:
    st.subheader("Enter Transaction Attributes")
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            tx_type = st.selectbox("Transaction Category", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=5000.0, step=100.0)
        with c2:
            st.write("**Balance Shift Preview**")
            diff_origin = st.empty()
            diff_dest = st.empty()

        st.divider()
        
        col_orig, col_dest = st.columns(2)
        with col_orig:
            st.markdown("##### 📤 Sender Account (Origin)")
            old_org = st.number_input("Pre-Transaction Balance", min_value=0.0, value=5000.0, step=100.0, key="old_org")
            new_org = st.number_input("Post-Transaction Balance", min_value=0.0, value=0.0, step=100.0, key="new_org")

        with col_dest:
            st.markdown("##### 📥 Receiver Account (Destination)")
            old_dest = st.number_input("Pre-Transaction Balance", min_value=0.0, value=0.0, step=100.0, key="old_dest")
            new_dest = st.number_input("Post-Transaction Balance", min_value=0.0, value=5000.0, step=100.0, key="new_dest")

        # Dynamic calculation display
        diff_origin.metric("Sender Net Change", f"${new_org - old_org:,.2f}")
        diff_dest.metric("Receiver Net Change", f"${new_dest - old_dest:,.2f}")

    if st.button("Run Risk Assessment", type="primary", use_container_width=True):
        if model is None:
            st.error("Please place the `fraud_detection_random_forest.pkl` file in the working directory.")
        else:
            raw_input = pd.DataFrame([{
                "type": tx_type,
                "amount": amount,
                "oldbalanceOrg": old_org,
                "newbalanceOrig": new_org,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest
            }])
            
            features = preprocess_input(raw_input)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None

            st.subheader("Assessment Result")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if probability is not None:
                    st.metric("Fraud Probability", f"{probability * 100:.1f}%")
                    st.progress(probability)

            with res_col2:
                if prediction == 1:
                    st.error("### 🚨 High Risk: Transaction Flagged as Fraudulent")
                    st.write("This transaction exhibits patterns consistent with fraudulent account draining or unauthorized transfers.")
                else:
                    st.success("### ✅ Low Risk: Transaction Verified Legitimate")
                    st.write("All balance shifts and amount profiles match standard user behavior patterns.")

# TAB 2: Batch CSV Prediction
with tab2:
    st.subheader("Upload Batch Records")
    uploaded_file = st.file_uploader("Upload CSV file containing transaction data", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview:", batch_df.head(5))
        
        if st.button("Process Batch Predictions"):
            if model is None:
                st.error("Model unavailable.")
            else:
                try:
                    processed_batch = preprocess_input(batch_df)
                    batch_df["Prediction"] = model.predict(processed_batch)
                    batch_df["Prediction_Label"] = batch_df["Prediction"].map({1: "FRAUD", 0: "LEGITIMATE"})
                    
                    if hasattr(model, "predict_proba"):
                        batch_df["Fraud_Probability"] = model.predict_proba(processed_batch)[:, 1].round(4)

                    st.divider()
                    st.subheader("Summary Report")
                    total = len(batch_df)
                    frauds = (batch_df["Prediction"] == 1).sum()
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Processed", total)
                    m2.metric("Flagged Fraud", frauds)
                    m3.metric("Fraud Rate", f"{(frauds/total)*100:.2f}%")

                    st.dataframe(batch_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")
