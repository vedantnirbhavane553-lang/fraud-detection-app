import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SentinelPay | Fraud Detection Operations",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (Clean FinTech Operations Theme) ---
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-fraud {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-clean {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADER ---
@st.cache_resource
def get_model():
    try:
        return joblib.load("fraud_detection_random_forest.pkl")
    except Exception:
        return None

model = get_model()

# --- SESSION STATE INITIALIZATION ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- HEADER BAR ---
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.title("⚡ SentinelPay Operations Console")
    st.caption("Real-Time Behavioral Fraud Engine & Transaction Risk Scoring")
with head_col2:
    st.markdown("<div style='text-align: right; margin-top: 15px;'>", unsafe_allow_html=True)
    if model is not None:
        st.success("🟢 ML Pipeline Active")
    else:
        st.error("🔴 Model Offline (Demo Mode)")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# --- 3-COLUMN DASHBOARD LAYOUT ---
col_input, col_viz, col_history = st.columns([1.2, 1.2, 1], gap="large")

# ==========================================
# COLUMN 1: INTERACTIVE TRANSACTION CREATOR
# ==========================================
with col_input:
    st.subheader("1. Transaction Form")
    
    with st.container(border=True):
        tx_type = st.selectbox(
            "Transaction Type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"],
            help="High-risk types typically include TRANSFER and CASH_OUT."
        )
        amount = st.number_input("Amount ($)", min_value=0.01, value=15000.0, step=500.0)
        
        st.markdown("**Origin Account (Sender)**")
        c1, c2 = st.columns(2)
        with c1:
            old_org = st.number_input("Old Balance", value=15000.0, step=500.0, key="orig_old")
        with c2:
            new_org = st.number_input("New Balance", value=0.0, step=500.0, key="orig_new")

        st.markdown("**Destination Account (Receiver)**")
        c3, c4 = st.columns(2)
        with c3:
            old_dest = st.number_input("Old Balance", value=0.0, step=500.0, key="dest_old")
        with c4:
            new_dest = st.number_input("New Balance", value=15000.0, step=500.0, key="dest_new")

        # Quick Math Audit
        org_diff = old_org - new_org
        discrepancy = org_diff - amount
        if abs(discrepancy) > 0.01 and tx_type in ["TRANSFER", "CASH_OUT"]:
            st.warning(f"⚠️ Sender balance delta (${org_diff:,.2f}) does not match amount (${amount:,.2f})")

        eval_btn = st.button("Score Transaction", type="primary", use_container_width=True)

# ==========================================
# COLUMN 2: RISK GAUGES & SCORECARD
# ==========================================
with col_viz:
    st.subheader("2. AI Risk Evaluation")
    
    # Feature Vector Preprocessing
    features = pd.DataFrame([{
        'step': 1,
        'amount': amount,
        'oldbalanceOrg': old_org,
        'newbalanceOrig': new_org,
        'oldbalanceDest': old_dest,
        'newbalanceDest': new_dest,
        'type_CASH_OUT': 1 if tx_type == "CASH_OUT" else 0,
        'type_DEBIT': 1 if tx_type == "DEBIT" else 0,
        'type_PAYMENT': 1 if tx_type == "PAYMENT" else 0,
        'type_TRANSFER': 1 if tx_type == "TRANSFER" else 0
    }])

    # Fallback simulation if model file is missing
    if model is not None:
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else (0.95 if prediction == 1 else 0.05)
    else:
        # Simple heuristic fallback for previewing UI without pkl
        prob = 0.92 if (tx_type in ["TRANSFER", "CASH_OUT"] and new_org == 0 and amount >= 10000) else 0.08
        prediction = 1 if prob >= 0.5 else 0

    # Record to history when button is clicked
    if eval_btn:
        st.session_state.history.insert(0, {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Type": tx_type,
            "Amount": f"${amount:,.0f}",
            "Score": f"{prob*100:.1f}%",
            "Verdict": "FRAUD" if prediction == 1 else "CLEAN"
        })

    # Plotly Gauge Indicator
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Risk Probability Index", 'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1e293b"},
            'steps': [
                {'range': [0, 30], 'color': "#86efac"},
                {'range': [30, 70], 'color': "#fde047"},
                {'range': [70, 100], 'color': "#fca5a5"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=220)
    st.plotly_chart(fig, use_container_width=True)

    # Decision Card
    with st.container(border=True):
        if prediction == 1:
            st.markdown('<span class="badge-fraud">⚠️ ACTION REQUIRED: FRAUD SUSPECTED</span>', unsafe_allow_html=True)
            st.markdown(f"**Policy Trigger:** Flagged by RF Classifier with **{prob*100:.1f}%** certainty.")
        else:
            st.markdown('<span class="badge-clean">✅ PASSED: TRANSACTION LEGITIMATE</span>', unsafe_allow_html=True)
            st.markdown(f"**Policy Trigger:** Clean behavioral signature (**{(1-prob)*100:.1f}%** normal).")

# ==========================================
# COLUMN 3: RUNNING AUDIT LOG / SESSION FEED
# ==========================================
with col_history:
    st.subheader("3. Audit Session Log")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            history_df,
            hide_index=True,
            use_container_width=True
        )
        if st.button("Clear Log", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No transactions scored in this session yet. Run a prediction to populate.")
