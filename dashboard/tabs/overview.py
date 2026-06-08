import streamlit as st


def render_overview(df_hist):
    st.title("🏠 Overview")
    st.markdown("**Credit Card Offer Acceptance Prediction Dashboard** — A research prototype for targeted marketing decision support.")
    st.divider()

    # ── Project Summary ──────────────────────────
    st.subheader("Project Summary")
    st.markdown("""
    This dashboard demonstrates how machine learning can support targeted credit card offer marketing by
    predicting which customers are more likely to accept a credit card offer. The dashboard is intended
    for financial institution marketing teams as a **decision-support prototype**.

    The final selected model is a **Threshold-tuned Logistic Regression** trained on historical customer
    and campaign data. The dashboard does not retrain the model — it loads the saved model files and
    applies them to new user-entered inputs.
    """)
    st.divider()

    # ── Key Project Metrics ──────────────────────
    st.subheader("Key Project Metrics")
    km1, km2, km3 = st.columns(3)
    km1.metric("Total Customers (Historical Dataset)", "18,000")
    km2.metric("Historical Accepted Offers", "1,023")
    km3.metric("Historical Acceptance Rate", "5.68%")
    km4, km5, km6 = st.columns(3)

    km4.metric(
        "Final Selected Model",
        "Logistic Regression",
        help="Threshold-tuned Logistic Regression was selected as the final model."
    )

    km5.metric("ROC-AUC Score", "0.764")

    km6.metric("Minority Class F1-Score", "0.26")
    st.divider()

    # ── Simulated Live Analytics ─────────────────
    st.subheader("Simulated Live Analytics", help="Updates only when predictions are made during the current dashboard session.")

    preds = st.session_state["session_predictions"]
    n_total  = len(preds)
    n_high   = sum(1 for p in preds if p["priority"] == "High Priority")
    n_medium = sum(1 for p in preds if p["priority"] == "Medium Priority")
    n_low    = sum(1 for p in preds if p["priority"] == "Low Priority")
    avg_prob = (sum(p["prob_pct"] for p in preds) / n_total) if n_total > 0 else None
    if n_total > 0:
        latest = preds[-1]["decision"]
        if "Not Recommended" in latest:
            latest_display = "Not Recommended"
        elif "Recommended" in latest:
            latest_display = "Recommended"
        else:
            latest_display = latest
    else:
        latest_display = "No predictions yet"

    la1, la2, la3 = st.columns(3)
    la1.metric("Customers Scored (This Session)", n_total)
    la2.metric("High-Priority Predictions", n_high)
    la3.metric("Medium-Priority Predictions", n_medium)
    la4, la5, la6 = st.columns(3)
    la4.metric("Low-Priority Predictions", n_low)
    la5.metric("Avg. Acceptance Probability", f"{avg_prob:.1f}%" if avg_prob is not None else "—")
    la6.metric("Latest Prediction Result", latest_display)

    st.info(
        "These live analytics are generated from predictions made during the current dashboard session. "
        "They are for prototype demonstration only and do not represent real deployed banking data."
    )
