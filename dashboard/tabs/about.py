import streamlit as st


def render_about():
    st.title("ℹ️ About")

    # Row 1 — Purpose | What This Dashboard is NOT
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<p class="section-header">Dashboard Purpose</p>', unsafe_allow_html=True)
            st.markdown("""
            This dashboard is a **prototype interface** that demonstrates how a trained machine learning
            model can support targeted credit card marketing decisions in a financial institution.

            Bank marketing officers can enter customer and campaign details, and receive a predicted
            **offer acceptance probability** along with a marketing priority recommendation.
            """)

    with col2:
        with st.container(border=True):
            st.markdown('<p class="section-header">What This Dashboard is NOT</p>', unsafe_allow_html=True)
            st.error("""
            * **Not** a credit approval system
            * Does **not** assess creditworthiness
            * Does **not** predict loan eligibility or default risk
            * Does **not** detect fraud

            It predicts whether a customer is likely to accept a credit card offer for marketing targeting purposes.
            """)

    # Row 2 — Intended Users | Limitations
    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown('<p class="section-header">Intended Users</p>', unsafe_allow_html=True)
            st.markdown("""
            - Bank marketing officers
            - Campaign decision-makers
            - Financial product marketing teams

            > This dashboard was also developed as an academic prototype for Final Year Project (FYP) demonstration purposes.
            > This dashboard is **not** designed for customers. It is an internal marketing support tool.
            """)

    with col4:
        with st.container(border=True):
            st.markdown('<p class="section-header">Limitations</p>', unsafe_allow_html=True)
            st.markdown("""
            - **Prototype only:** Not intended for live production banking use.
            - **Historical data:** Predictions are based on patterns from past customer data and may not generalise to all future scenarios.
            - **Imbalanced dataset:** The accepted class is a minority; the model is optimised for recall at the cost of precision.
            - **No real-time data:** The dashboard uses manual input and a saved trained model. It is not connected to live banking databases or customer records.
            - **Academic context:** This dashboard was developed as part of a Final Year Project (FYP) research study.
            """)

    # Row 3 — Dataset (full-width card)
    with st.container(border=True):
        st.markdown('<p class="section-header">Dataset</p>', unsafe_allow_html=True)
        st.markdown("""
        - **Name:** Credit Card Offer Acceptance Trends in Banking
        - **Target variable:** Whether a customer accepted a credit card offer (`1` = Accepted, `0` = Not Accepted)
        - **Train / Test split:** 80% training, 20% testing
        - **Class imbalance handling:** The model was adjusted to give more attention to customers who accepted the offer, since this group is much smaller in the dataset (implemented using `class_weight='balanced'` in Logistic Regression)
        """)

    st.divider()
    st.caption(
        "Disclaimer: This dashboard is developed for academic purposes as part of a Final Year Project. "
        "Predictions are based on a prototype machine learning model and should not be used for actual credit decisions."
    )
