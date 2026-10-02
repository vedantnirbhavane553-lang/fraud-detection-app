"""Fraud Detection Prediction App.

Run locally:
    python -m streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import build_features

BASE = Path(__file__).parent
NEW_MODEL, NEW_META = BASE / "fraud_model.pkl", BASE / "model_meta.json"
OLD_MODEL = BASE / "fraud_detection_random_forest.pkl"
LEGACY_THRESHOLD = 0.928  # from the original notebook (it was tuned on the test set)

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the model once per server process, not on every interaction."""
    if NEW_MODEL.exists():
        meta = json.loads(NEW_META.read_text()) if NEW_META.exists() else {}
        return joblib.load(NEW_MODEL), meta.get("threshold", LEGACY_THRESHOLD), meta, True
    if OLD_MODEL.exists():
        return joblib.load(OLD_MODEL), LEGACY_THRESHOLD, {}, False
    return None, None, {}, False


model, default_threshold, meta, validated = load_model()
if model is None:
    st.error("No model file found. Run `python train_model.py --data <csv>` or add the .pkl file.")
    st.stop()

st.title("Fraud Detection Prediction App")
st.write("Enter the transaction details to get a fraud risk score.")

with st.sidebar:
    st.header("Decision threshold")
    threshold = st.slider(
        "Flag as fraud when risk is at least",
        0.05, 0.99, float(round(default_threshold, 3)), 0.01,
        help="Lower = catch more fraud but more false alarms. Higher = fewer false alarms but more missed fraud.",
    )
    if validated:
        st.caption(f"Model: {meta.get('model', 'unknown')}. Threshold tuned on a validation set.")
    else:
        st.caption(
            "Legacy model: its default threshold was tuned on the test set, so its "
            "reported scores are optimistic. Retrain with `train_model.py` for a validated one."
        )
    st.caption("Trained on PaySim, a synthetic dataset. Scores here do not reflect real-world accuracy.")

with st.form("transaction"):
    c1, c2 = st.columns(2)
    transaction_type = c1.selectbox("Transaction type", TRANSACTION_TYPES)
    step = c2.number_input("Hour of simulation (step)", min_value=1, max_value=743, value=1, step=1,
                           help="PaySim runs for 743 hourly steps.")
    amount = c1.number_input("Amount", min_value=0.0, value=1000.0)
    old_org = c2.number_input("Sender balance before", min_value=0.0, value=0.0)
    new_org = c1.number_input("Sender balance after", min_value=0.0, value=0.0)
    old_dest = c2.number_input("Receiver balance before", min_value=0.0, value=0.0)
    new_dest = c1.number_input("Receiver balance after", min_value=0.0, value=0.0)
    submitted = st.form_submit_button("Check transaction", type="primary")

if submitted:
    raw = pd.DataFrame([{
        "step": step, "type": transaction_type, "amount": amount,
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
    }])
    features = build_features(raw)
    # Use the exact columns, in the exact order, the model was trained on.
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        features = features.reindex(columns=list(expected), fill_value=0)

    probability = float(model.predict_proba(features)[0, 1])
    is_fraud = probability >= threshold

    st.divider()
    left, right = st.columns([1, 2])
    left.metric("Fraud risk", f"{probability:.1%}")
    right.progress(min(probability, 1.0), text=f"Flag threshold: {threshold:.0%}")

    if is_fraud:
        st.error(f"This transaction is flagged as **likely fraud** ({probability:.1%} risk).")
    else:
        st.success(f"This transaction looks **legitimate** ({probability:.1%} risk).")

    notes = []
    if transaction_type in ("PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT") and abs(old_org - amount - new_org) > 0.01:
        notes.append("Sender balances don't add up (before - amount != after).")
    if old_org > 0 and new_org == 0:
        notes.append("The sender's account was emptied.")
    if transaction_type in ("CASH_IN", "DEBIT", "PAYMENT") and amount == 0:
        notes.append("The amount is zero.")
    if notes:
        with st.expander("Signals worth a look"):
            for n in notes:
                st.write("- " + n)

    with st.expander("Features sent to the model"):
        st.dataframe(features.T.rename(columns={0: "value"}), use_container_width=True)
