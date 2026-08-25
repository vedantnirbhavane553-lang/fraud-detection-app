import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Model Benchmark | Fraud Ensemble",
    page_icon="⚖️",
    layout="wide"
)

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_all_models():
    models = {}
    model_files = {
        "Random Forest": "fraud_detection_random_forest.pkl",
        "XGBoost": "fraud_detection_xgboost.pkl",
        "Logistic Regression": "fraud_detection_logistic_regression.pkl"
    }
    
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except Exception:
            models[name] = None
    return models

loaded_models = load_all_models()

# --- HEADER SECTION ---
st.title("⚖️ Multi-Model Fraud Consensus & Benchmarking")
st.caption("Compare individual classifier predictions, probabilities, and model consensus in real time.")

# Model status ribbon
st.markdown("##### Model Pipeline Status")
st_cols = st.columns(3)
for col, (m_name, m_obj) in zip(st_cols, loaded_models.items()):
    with col:
        if m_obj is not None:
            st.success(f"🟢 **{m_name}**: Online")
        else:
            st.warning(f"🟡 **{m_name}**: Mock Simulation")

st.divider()

# --- LAYOUT: INPUT FORM & COMPARISON RESULTS ---
col_input, col_results = st.columns([1, 1.4], gap="large")

# ==========================================
# LEFT PANEL: TRANSACTION INPUTS
# ==========================================
with col_input:
    st.subheader("1. Transaction Parameters")
    
    with st.container(border=True):
        tx_type = st.selectbox(
            "Transaction Type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"],
            index=0
        )
        amount = st.number_input("Amount ($)", min_value=0.01, value=125000.0, step=1000.0)
        
        st.markdown("**Origin Node (Sender)**")
        c1, c2 = st.columns(2)
        with c1:
            old_org = st.number_input("Sender Old Balance", value=125000.0, step=1000.0)
        with c2:
            new_org = st.number_input("Sender New Balance", value=0.0, step=1000.0)

        st.markdown("**Target Node (Receiver)**")
        c3, c4 = st.columns(2)
        with c3:
            old_dest = st.number_input("Receiver Old Balance", value=0.0, step=1000.0)
        with c4:
            new_dest = st.number_input("Receiver New Balance", value=125000.0, step=1000.0)

    # Threshold config
    with st.expander("⚙️ Decision Thresholds & Weights"):
        fraud_threshold = st.slider("Classification Risk Cutoff", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        st.caption("Transactions with an ensemble probability above this threshold are classified as FRAUD.")

# ==========================================
# RIGHT PANEL: MODEL BENCHMARK & CONSENSUS
# ==========================================
with col_results:
    st.subheader("2. Multi-Model Benchmark")

    # Construct input feature row
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

    # Collect individual model outputs
    results = {}
    
    # Mock behavioral fallback function if specific model files are absent
    def mock_predict(model_name, features_df):
        # Heuristic simulation for previewing when pkl files are missing
        is_drained = (old_org > 0 and new_org == 0)
        is_transfer = (tx_type in ["TRANSFER", "CASH_OUT"])
        
        if model_name == "Random Forest":
            return (0.94 if (is_transfer and is_drained) else 0.04)
        elif model_name == "XGBoost":
            return (0.98 if (is_transfer and is_drained) else 0.02)
        else: # Logistic Regression
            return (0.82 if (is_transfer and is_drained) else 0.12)

    for name, m_obj in loaded_models.items():
        if m_obj is not None:
            if hasattr(m_obj, "predict_proba"):
                prob = m_obj.predict_proba(features)[0][1]
            else:
                prob = float(m_obj.predict(features)[0])
        else:
            prob = mock_predict(name, features)
        
        pred = 1 if prob >= fraud_threshold else 0
        results[name] = {"probability": prob, "prediction": pred}

    # Display side-by-side metric cards
    m_col1, m_col2, m_col3 = st.columns(3)
    card_cols = [m_col1, m_col2, m_col3]

    for col, (name, res) in zip(card_cols, results.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                score_pct = res["probability"] * 100
                st.metric("Fraud Risk", f"{score_pct:.1f}%")
                if res["prediction"] == 1:
                    st.error("🚨 FRAUD")
                else:
                    st.success("✅ CLEAN")

    # Calculate Ensemble Metrics
    avg_probability = np.mean([r["probability"] for r in results.values()])
    fraud_votes = sum([r["prediction"] for r in results.values()])
    total_models = len(results)

    st.markdown("---")
    
    # Consensus summary box
    with st.container(border=True):
        c_left, c_right = st.columns([1.5, 1])
        with c_left:
            st.markdown("#### 🛡️ Ensemble Consensus Verdict")
            if fraud_votes >= 2: # Majority vote (2 out of 3)
                st.error(f"### 🚨 HIGH RISK: FLAGGED AS FRAUD ({fraud_votes}/{total_models} Models Agreed)")
                st.write(f"Mean ensemble risk score is **{avg_probability*100:.1f}%**, exceeding the safety threshold.")
            else:
                st.success(f"### ✅ LOW RISK: TRANSACTION APPROVED ({total_models - fraud_votes}/{total_models} Models Agreed)")
                st.write(f"Mean ensemble risk score is **{avg_probability*100:.1f}%**, within standard limits.")

        with c_right:
            st.metric("Mean Ensemble Probability", f"{avg_probability * 100:.1f}%")
            st.progress(float(avg_probability))

    # Comparative Plotly Bar Chart
    model_names = list(results.keys())
    probabilities = [results[m]["probability"] * 100 for m in model_names]
    bar_colors = ['#ef4444' if p >= (fraud_threshold * 100) else '#10b981' for p in probabilities]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_names,
        y=probabilities,
        text=[f"{p:.1f}%" for p in probabilities],
        textposition='auto',
        marker_color=bar_colors,
        width=0.4
    ))

    fig.add_hline(
        y=fraud_threshold * 100, 
        line_dash="dash", 
        line_color="#f59e0b",
        annotation_text=f"Cutoff Threshold ({fraud_threshold*100:.0f}%)",
        annotation_position="bottom right"
    )

    fig.update_layout(
        title="Model Probability Comparison",
        yaxis=dict(title="Fraud Probability (%)", range=[0, 100]),
        xaxis=dict(title=""),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
