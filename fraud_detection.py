import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CyberRisk Matrix | Anti-Fraud Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DARK CYBER THEME CSS ---
st.markdown("""
<style>
    /* Global Dark Theme Adjustments */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Cyber Card Container */
    .cyber-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Terminal Console Display */
    .terminal-box {
        background-color: #030712;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.85rem;
        color: #10b981;
    }
    
    /* Status Badges */
    .status-pill-red {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .status-pill-green {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADER ---
@st.cache_resource
def load_fraud_model():
    try:
        return joblib.load("fraud_detection_random_forest.pkl")
    except Exception:
        return None

model = load_fraud_model()

# --- PRESET SCENARIOS ---
SCENARIOS = {
    "Manual Input": None,
    "🚨 Account Drain Attack (Transfer)": {
        "type": "TRANSFER",
        "amount": 250000.0,
        "old_org": 250000.0,
        "new_org": 0.0,
        "old_dest": 0.0,
        "new_dest": 250000.0
    },
    "🚨 Rapid Cash-Out Exploit": {
        "type": "CASH_OUT",
        "amount": 85000.0,
        "old_org": 85000.0,
        "new_org": 0.0,
        "old_dest": 1200.0,
        "new_dest": 86200.0
    },
    "✅ Standard Merchant Payment": {
        "type": "PAYMENT",
        "amount": 120.50,
        "old_org": 4200.0,
        "new_org": 4079.50,
        "old_dest": 0.0,
        "new_dest": 0.0
    },
    "✅ Standard Account Cash-In": {
        "type": "CASH_IN",
        "amount": 1500.0,
        "old_org": 500.0,
        "new_org": 2000.0,
        "old_dest": 20000.0,
        "new_dest": 18500.0
    }
}

# --- HEADER ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("## 🛡️ THREAT-INTEL // FRAUD INVESTIGATION MATRIX")
    st.caption("Deep-packet behavioral analysis & automated anomaly classification")
with col_head2:
    if model is not None:
        st.markdown("<span class='status-pill-green'>● SYSTEM ONLINE</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-pill-red'>▲ MODEL OFFLINE (SIMULATION)</span>", unsafe_allow_html=True)

st.markdown("---")

# --- SCENARIO SELECTOR PRESET BAR ---
st.markdown("##### ⚡ Quick-Load Test Vectors")
preset_choice = st.selectbox(
    "Select a pre-configured behavioral profile to autofill values:",
    options=list(SCENARIOS.keys())
)

current_scenario = SCENARIOS[preset_choice]

# Default values based on preset
default_type = current_scenario["type"] if current_scenario else "TRANSFER"
default_amt = current_scenario["amount"] if current_scenario else 50000.0
default_old_org = current_scenario["old_org"] if current_scenario else 50000.0
default_new_org = current_scenario["new_org"] if current_scenario else 0.0
default_old_dest = current_scenario["old_dest"] if current_scenario else 0.0
default_new_dest = current_scenario["new_dest"] if current_scenario else 50000.0

# --- MAIN INVESTIGATION CONSOLE ---
col_left, col_right = st.columns([1.1, 1], gap="medium")

# LEFT PANEL: Parameters
with col_left:
    st.markdown("### 📥 Transaction Telemetry")
    
    with st.container(border=True):
        t1, t2 = st.columns([1, 1])
        with t1:
            type_options = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
            type_idx = type_options.index(default_type) if default_type in type_options else 0
            tx_type = st.selectbox("Protocol (Type)", type_options, index=type_idx)
        with t2:
            amount = st.number_input("Payload Amount ($)", min_value=0.01, value=float(default_amt), step=1000.0)

        st.markdown("##### Origin Node (Sender Ledger)")
        o1, o2 = st.columns(2)
        with o1:
            old_org = st.number_input("Pre-Balance", value=float(default_old_org), step=1000.0, key="orig_old_val")
        with o2:
            new_org = st.number_input("Post-Balance", value=float(default_new_org), step=1000.0, key="orig_new_val")

        st.markdown("##### Target Node (Destination Ledger)")
        d1, d2 = st.columns(2)
        with d1:
            old_dest = st.number_input("Pre-Balance", value=float(default_old_dest), step=1000.0, key="dest_old_val")
        with d2:
            new_dest = st.number_input("Post-Balance", value=float(default_new_dest), step=1000.0, key="dest_new_val")

    run_analysis = st.button("RUN FORENSIC CLASSIFIER", type="primary", use_container_width=True)

# RIGHT PANEL: Evaluation & Risk Breakdown
with col_right:
    st.markdown("### 🧠 Threat Evaluation Matrix")
    
    # Construct input dataframe
    input_vector = pd.DataFrame([{
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

    # Model inference or fallback heuristic
    if model is not None:
        prediction = model.predict(input_vector)[0]
        prob = model.predict_proba(input_vector)[0][1] if hasattr(model, "predict_proba") else (0.95 if prediction == 1 else 0.05)
    else:
        # Behavioral heuristic logic for demonstration
        is_high_risk_type = tx_type in ["TRANSFER", "CASH_OUT"]
        is_drained = (old_org > 0 and new_org == 0)
        prob = 0.94 if (is_high_risk_type and is_drained) else 0.04
        prediction = 1 if prob >= 0.50 else 0

    with st.container(border=True):
        res_header, res_badge = st.columns([2, 1])
        with res_header:
            st.markdown(f"**Anomaly Confidence Score:** `{prob * 100:.2f}%`")
        with res_badge:
            if prediction == 1:
                st.markdown("<span class='status-pill-red'>THREAT DETECTED</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='status-pill-green'>CLEAN SIGNATURE</span>", unsafe_allow_html=True)

        st.progress(float(prob))

        # Risk Vector Heuristic Breakdown
        st.markdown("##### 🔬 Signal Attribution Checklist")
        c_drain = "🔴" if (old_org > 0 and new_org == 0) else "🟢"
        c_type = "🔴" if tx_type in ["TRANSFER", "CASH_OUT"] else "🟢"
        c_amt = "🟡" if amount > 50000 else "🟢"

        st.markdown(f"{c_type} **Transaction Vector:** `{tx_type}` {'(High Risk Protocol)' if tx_type in ['TRANSFER', 'CASH_OUT'] else '(Standard Protocol)'}")
        st.markdown(f"{c_drain} **Sender Liquidation:** {'Account 100% drained to 0 balance' if (old_org > 0 and new_org == 0) else 'Normal account balance delta'}")
        st.markdown(f"{c_amt} **Exposure Magnitude:** ${amount:,.2f}")

    # Collapsible Raw Inspection Box
    with st.expander("🛠️ Raw Telemetry JSON Payload"):
        st.code(json.dumps(input_vector.to_dict(orient="records")[0], indent=2), language="json")
