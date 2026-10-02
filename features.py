"""Feature engineering shared by train_model.py and app.py.

Keeping this in one place prevents train/serve skew: the app builds
exactly the same columns the model was trained on.
"""

import pandas as pd

# CASH_IN is the baseline category (all type_* columns = 0), as with
# pd.get_dummies(..., drop_first=True) in the original notebook.
TYPE_COLUMNS = ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw PaySim-style columns into model features.

    Required columns: step, type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest.
    """
    out = pd.DataFrame(index=df.index)
    out["step"] = df["step"]
    out["amount"] = df["amount"]
    out["oldbalanceOrg"] = df["oldbalanceOrg"]
    out["newbalanceOrig"] = df["newbalanceOrig"]
    out["oldbalanceDest"] = df["oldbalanceDest"]
    out["newbalanceDest"] = df["newbalanceDest"]

    for t in TYPE_COLUMNS:
        out[f"type_{t}"] = (df["type"] == t).astype(int)

    # Engineered signals: in clean transactions the balances add up.
    out["balance_error_orig"] = out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
    out["balance_error_dest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    out["orig_emptied"] = ((out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)).astype(int)
    return out
