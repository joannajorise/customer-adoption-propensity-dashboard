import streamlit as st


def render_overview(df_hist):
    # Hero / Header Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-badge">Research Prototype</div>
            <h1 class="hero-title">Predicting Customer Adoption Propensity Using Spending Behaviour</h1>
            <p class="hero-subtitle">A decision-support prototype powered by machine learning for targeting credit card offer campaigns.</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Project Summary ──────────────────────────
    with st.container(border=True):
        st.markdown("""
            <div style="margin: 0.5rem 0.75rem;">
                <h3 style="margin-top: 0; color: #1e2d4a; font-size: 1.15rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
                    Project Summary
                </h3>
                <p style="font-size: 0.92rem; color: #4b5563; line-height: 1.65; margin-bottom: 0;">
                    This dashboard demonstrates how machine learning can support targeted credit card offer marketing by
                    predicting which customers are more likely to accept a credit card offer. The dashboard is intended
                    for financial institution marketing teams as a <strong>decision-support prototype</strong>.
                </p>
                <p style="font-size: 0.92rem; color: #4b5563; line-height: 1.65; margin-top: 0.75rem; margin-bottom: 0;">
                    The final selected model is a <strong>Threshold-tuned Logistic Regression</strong> trained on historical customer
                    and campaign data. The dashboard does not retrain the model — it loads the saved model files and
                    applies them to new user-entered inputs.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ── Key Project Metrics ──────────────────────
    st.markdown('<p class="section-header" style="margin-top: 2rem;">Key Project Metrics</p>', unsafe_allow_html=True)
    
    col_data, col_model = st.columns(2)
    
    with col_data:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; color: #2c4064; margin-bottom: 0.75rem; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">Historical Dataset</div>', unsafe_allow_html=True)
            km1, km2, km3 = st.columns(3)
            km1.metric("Total Customers", "18,000")
            km2.metric("Accepted Offers", "1,023")
            km3.metric("Acceptance Rate", "5.68%")
            
    with col_model:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; color: #2c4064; margin-bottom: 0.75rem; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">Final Predictive Model</div>', unsafe_allow_html=True)
            km4, km5, km6 = st.columns(3)
            km4.metric("Selected Model", "Logistic Reg.", help="Threshold-tuned Logistic Regression was selected as the final model.")
            km5.metric("ROC-AUC Score", "0.764")
            km6.metric("F1-Score", "0.26", help="Minority Class F1-Score")

    # ── Simulated Live Analytics ─────────────────
    st.markdown('<p class="section-header" style="margin-top: 2rem;">Simulated Live Analytics</p>', unsafe_allow_html=True)

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

    # Layout for Simulated Live Analytics
    col_session, col_dist = st.columns([5, 4])
    
    with col_session:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; color: #2c4064; margin-bottom: 0.75rem; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">Session Activity</div>', unsafe_allow_html=True)
            la1, la2, la3 = st.columns(3)
            la1.metric("Scored Customers", n_total)
            la2.metric("Avg. Propensity", f"{avg_prob:.1f}%" if avg_prob is not None else "—")
            la3.metric("Latest Prediction", latest_display)
            
    with col_dist:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; color: #2c4064; margin-bottom: 0.75rem; font-size: 0.95rem; display: flex; align-items: center; gap: 6px;">Priority Distribution</div>', unsafe_allow_html=True)
            pd1, pd2, pd3 = st.columns(3)
            pd1.metric("🔴 High", n_high)
            pd2.metric("🟡 Medium", n_medium)
            pd3.metric("⚪ Low", n_low)

    st.markdown(" ")
    st.info(
        "These live analytics are generated from predictions made during the current dashboard session. "
        "They are for prototype demonstration only and do not represent real deployed banking data."
    )
