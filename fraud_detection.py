
def score_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Apply an intentionally simple, explainable demo risk score (0–100)."""
    frame = data.copy()
    for col, default in {"amount": 0, "device_new": False, "failed_attempts": 0, "country": "Unknown"}.items():
        if col not in frame:
            frame[col] = default
    amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0)
    score = (amount > 250).astype(int) * 28 + (amount > 750).astype(int) * 20
    score += frame["device_new"].astype(bool).astype(int) * 22
    score += pd.to_numeric(frame["failed_attempts"], errors="coerce").fillna(0).clip(0, 4) * 9
    score += frame["country"].isin(["United States", "UAE"]).astype(int) * 8
    frame["risk_score"] = score.clip(0, 100).astype(int)
    frame["risk_level"] = pd.cut(
        frame["risk_score"], bins=[-1, 29, 59, 100], labels=["Low", "Medium", "High"]
    ).astype(str)
    return frame


def metric_card(label: str, value: str, delta: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-delta">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def load_transactions(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return sample_transactions()
    raw = pd.read_csv(uploaded_file)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    if "transaction_id" not in raw:
        raw["transaction_id"] = [f"UPLOAD-{i + 1}" for i in range(len(raw))]
    return score_transactions(raw)


inject_css()

with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.caption("INTELLIGENCE CONSOLE")
    st.divider()
    page = st.radio("Navigate", ["Overview", "Transaction review", "Risk insights"], label_visibility="collapsed")
    st.divider()
    uploaded = st.file_uploader("Upload transactions (.csv)", type="csv")
    st.caption("Expected fields: amount, country, device_new, failed_attempts.")
    st.divider()
    st.caption("MODEL STATUS")
    st.success("Monitoring active")

transactions = load_transactions(uploaded)
high_risk = transactions[transactions.risk_level == "High"]
total_amount = pd.to_numeric(transactions.amount, errors="coerce").fillna(0).sum()

st.markdown(
    '<div class="hero"><h1>Fraud operations, made clear.</h1>'
    '<p>Review risk signals, prioritize alerts, and protect every transaction.</p></div>',
    unsafe_allow_html=True,
)

if page == "Overview":
    a, b, c, d = st.columns(4)
    with a: metric_card("Transactions monitored", f"{len(transactions):,}", "Live demo feed")
    with b: metric_card("High-risk alerts", f"{len(high_risk):,}", "Needs review")
    with c: metric_card("Exposure flagged", f"${high_risk.amount.sum():,.0f}", "Potentially at risk")
    with d: metric_card("Total volume", f"${total_amount:,.0f}", "All transactions")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="section-title">Risk activity</div>', unsafe_allow_html=True)
        activity = transactions.copy()
        activity["date"] = pd.to_datetime(activity.get("timestamp", pd.Timestamp.now())).dt.date
        chart = activity.groupby(["date", "risk_level"]).size().unstack(fill_value=0)
        st.area_chart(chart, color=["#7dd3a8", "#f5c95c", "#ec6b6b"])
    with right:
        st.markdown('<div class="section-title">Risk distribution</div>', unsafe_allow_html=True)
        counts = transactions.risk_level.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        st.bar_chart(counts, color="#315f90")

    st.markdown('<div class="section-title">Priority alerts</div>', unsafe_allow_html=True)
    view = high_risk.sort_values("risk_score", ascending=False).head(8)
    st.dataframe(view[["transaction_id", "merchant", "amount", "country", "risk_score", "risk_level"]], hide_index=True, use_container_width=True)

elif page == "Transaction review":
    st.markdown('<div class="section-title">Transaction review queue</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1: level = st.multiselect("Risk level", ["Low", "Medium", "High"], default=["High", "Medium"])
    with f2: minimum = st.number_input("Minimum score", 0, 100, 30)
    with f3: query = st.text_input("Search merchant, country, or transaction ID")
    reviewed = transactions[transactions.risk_level.isin(level) & (transactions.risk_score >= minimum)].copy()
    if query:
        mask = reviewed.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        reviewed = reviewed[mask]
    st.dataframe(reviewed.sort_values("risk_score", ascending=False), hide_index=True, use_container_width=True, height=520)
    st.download_button("Download filtered results", reviewed.to_csv(index=False), "fraud_review_queue.csv", "text/csv")

else:
    st.markdown('<div class="section-title">What drives risk?</div>', unsafe_allow_html=True)
    x, y = st.columns([1.1, 1])
    with x:
        st.markdown("""**Demo scoring signals**

        - Large transaction amounts
        - First-time device usage
        - Repeated failed attempts
        - Selected cross-border activity

        The app exposes these signals so every review decision can be understood.""")
    with y:
        signals = pd.DataFrame({"Signal": ["Large amount", "New device", "Failed attempts", "Cross-border"], "Weight": [48, 22, 36, 8]})
        st.bar_chart(signals.set_index("Signal"), color="#e9814a")
    st.info("This dashboard uses a transparent rules-based score for demonstration. Replace `score_transactions()` with your trained model when it is ready.")
