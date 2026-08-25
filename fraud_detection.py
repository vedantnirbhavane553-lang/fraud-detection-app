import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Fraud Desk | Analyst Case Manager",
    page_icon="💼",
    layout="wide"
)

# --- LOAD MODEL SAFELY ---
@st.cache_resource
def load_fraud_model():
    try:
        return joblib.load("fraud_detection_random_forest.pkl")
    except Exception:
        return None

model = load_fraud_model()

# --- INITIALIZE MOCK CASE QUEUE ---
if "cases" not in st.session_state:
    st.session_state.cases = [
        {
            "case_id": "TX-9021",
            "timestamp": "2026-08-25 14:22:10",
            "type": "TRANSFER",
            "amount": 185000.0,
            "oldbalanceOrg": 185000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 1200.0,
            "newbalanceDest": 186200.0,
            "status": "PENDING REVIEW"
        },
        {
            "case_id": "TX-9022",
            "timestamp": "2026-08-25 14:25:44",
            "type": "PAYMENT",
            "amount": 420.0,
            "oldbalanceOrg": 5200.0,
            "newbalanceOrig": 4780.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "status": "AUTO-APPROVED"
        },
        {
            "case_id": "TX-9023",
            "timestamp": "2026-08-25 14:29:01",
            "type": "CASH_OUT",
            "amount": 94000.0,
            "oldbalanceOrg": 94000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 450.0,
            "newbalanceDest": 94450.0,
            "status": "PENDING REVIEW"
        }
    ]

# --- APP HEADER ---
st.title("💼 Analyst Workbench: Case Review & Action Desk")
st.caption("Investigate flagged transactions, document findings, and execute risk-mitigation actions.")
st.divider()

# --- 2-COLUMN LAYOUT: INCOMING QUEUE & ACTIVE INVESTIGATION ---
col_queue, col_detail = st.columns([1, 1.6], gap="large")

# ==========================================
# LEFT COLUMN: CASE QUEUE SELECTOR
# ==========================================
with col_queue:
    st.subheader("📋 Triage Queue")
    
    # Render mini queue selector
    case_ids = [c["case_id"] for c in st.session_state.cases]
    selected_case_id = st.radio(
        "Select Case to Investigate:",
        options=case_ids,
        format_func=lambda x: f"{x} — {[c['type'] for c in st.session_state.cases if c['case_id'] == x][0]} (${[c['amount'] for c in st.session_state.cases if c['case_id'] == x][0]:,.0f})"
    )
    
    # Retrieve current active case object
    active_case = next(c for c in st.session_state.cases if c["case_id"] == selected_case_id)
    
    with st.container(border=True):
        st.markdown(f"**Case ID:** `{active_case['case_id']}`")
        st.markdown(f"**Status:** `{active_case['status']}`")
        st.markdown(f"**Ingested At:** `{active_case['timestamp']}`")

    # Fast New Case Injection Form
    with st.expander("➕ Inject Custom Test Case"):
        new_id = f"TX-{len(st.session_state.cases) + 9021}"
        new_type = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"], key="inj_type")
        new_amt = st.number_input("Amount", value=10000.0, key="inj_amt")
        if st.button("Add to Queue", use_container_width=True):
            st.session_state.cases.append({
                "case_id": new_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": new_type,
                "amount": new_amt,
                "oldbalanceOrg": new_amt,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": new_amt,
                "status": "PENDING REVIEW"
            })
            st.rerun()

# ==========================================
# RIGHT COLUMN: ACTIVE INVESTIGATION WORKBENCH
# ==========================================
with col_detail:
    st.subheader(f"🔍 Case Details: {active_case['case_id']}")
    
    # Preprocess selected case features
    input_features = pd.DataFrame([{
        'step': 1,
        'amount': active_case['amount'],
        'oldbalanceOrg': active_case['oldbalanceOrg'],
        'newbalanceOrig': active_case['newbalanceOrig'],
        'oldbalanceDest': active_case['oldbalanceDest'],
        'newbalanceDest': active_case['newbalanceDest'],
        'type_CASH_OUT': 1 if active_case['type'] == "CASH_OUT" else 0,
        'type_DEBIT': 1 if active_case['type'] == "DEBIT" else 0,
        'type_PAYMENT': 1 if active_case['type'] == "PAYMENT" else 0,
        'type_TRANSFER': 1 if active_case['type'] == "TRANSFER" else 0
    }])

    # Run AI inference
    if model is not None:
        prob = model.predict_proba(input_features)[0][1] if hasattr(model, "predict_proba") else 0.5
    else:
        # Calibrated heuristic fallback
        is_high_risk = active_case['type'] in ["TRANSFER", "CASH_OUT"] and active_case['newbalanceOrig'] == 0
        prob = 0.96 if is_high_risk else 0.03

    # Display Live Risk Scorecard
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Transaction Volume", f"${active_case['amount']:,.2f}")
        m2.metric("Protocol Category", active_case['type'])
        m3.metric("AI Risk Score", f"{prob * 100:.1f}%")
        
        st.progress(float(prob))
        if prob > 0.70:
            st.error("⚠️ Threat Alert: Severe anomaly detected (Complete sender account liquidation pattern).")
        else:
            st.success("✅ Clean Signature: Balance velocity aligns with standard behavior.")

    # Ledger Snapshot
    st.markdown("##### Account Ledger Reconstruction")
    l1, l2 = st.columns(2)
    with l1:
        st.caption("Sender (Origin Node)")
        st.dataframe(pd.DataFrame({
            "Metric": ["Pre-Balance", "Post-Balance", "Net Flow"],
            "Amount": [
                f"${active_case['oldbalanceOrg']:,.2f}",
                f"${active_case['newbalanceOrig']:,.2f}",
                f"-${active_case['oldbalanceOrg'] - active_case['newbalanceOrig']:,.2f}"
            ]
        }), hide_index=True, use_container_width=True)

    with l2:
        st.caption("Receiver (Target Node)")
        st.dataframe(pd.DataFrame({
            "Metric": ["Pre-Balance", "Post-Balance", "Net Flow"],
            "Amount": [
                f"${active_case['oldbalanceDest']:,.2f}",
                f"${active_case['newbalanceDest']:,.2f}",
                f"+${active_case['newbalanceDest'] - active_case['oldbalanceDest']:,.2f}"
            ]
        }), hide_index=True, use_container_width=True)

    # Analyst Action Form
    st.markdown("##### ✍️ Analyst Disposition & Case Resolution")
    notes = st.text_area("Investigation Notes", placeholder="Enter root cause analysis, phone verification notes, or suspicious IP details...")
    
    act1, act2, act3 = st.columns(3)
    with act1:
        if st.button("✅ Approve Transaction", use_container_width=True):
            active_case["status"] = "MANUALLY APPROVED"
            st.success(f"Case {active_case['case_id']} approved.")
            st.rerun()
    with act2:
        if st.button("🚫 Freeze & Flag Fraud", type="primary", use_container_width=True):
            active_case["status"] = "CONFIRMED FRAUD / FROZEN"
            st.error(f"Case {active_case['case_id']} marked as FRAUD.")
            st.rerun()
    with act3:
        if st.button("⚠️ Escalate to Tier 3", use_container_width=True):
            active_case["status"] = "ESCALATED TO T3"
            st.warning(f"Case {active_case['case_id']} escalated.")
            st.rerun()

    # CSV / Report Export
    report_data = {
        "Case ID": [active_case["case_id"]],
        "Timestamp": [active_case["timestamp"]],
        "Risk Score": [f"{prob*100:.2f}%"],
        "Final Status": [active_case["status"]],
        "Analyst Notes": [notes if notes else "N/A"]
    }
    report_df = pd.DataFrame(report_data)
    st.download_button(
        label="📥 Export Investigation Audit Report (CSV)",
        data=report_df.to_csv(index=False),
        file_name=f"audit_report_{active_case['case_id']}.csv",
        mime="text/csv",
        use_container_width=True
    )
