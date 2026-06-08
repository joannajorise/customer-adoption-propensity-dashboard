import streamlit as st
import pandas as pd


def render_prediction(model, scaler, model_columns, best_threshold, reset_form):
    st.title("🔍 Prediction")
    st.markdown("Enter customer and campaign details to receive an offer acceptance prediction.")
    st.divider()

    st.markdown("### Customer & Campaign Details")

    # Group 1 — Campaign Details
    st.markdown('<p class="section-label">Group 1 — Campaign Details</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        reward_options = ["Air Miles", "Cash Back", "Points"]
        reward_idx = reward_options.index(st.session_state.reward) if st.session_state.reward in reward_options else 0
        reward = st.selectbox(
            "Reward",
            reward_options,
            index=reward_idx,
            key="reward",
            help="Type of reward offered in the campaign."
        )
    with col2:
        mailer_options = ["Letter", "Postcard"]
        mailer_idx = mailer_options.index(st.session_state.mailer_type) if st.session_state.mailer_type in mailer_options else 0
        mailer_type = st.selectbox(
            "Mailer Type",
            mailer_options,
            index=mailer_idx,
            key="mailer_type",
            help="Format used to send the credit card offer."
        )

    st.markdown("---")

    # Group 2 — Customer Profile
    st.markdown('<p class="section-label">Group 2 — Customer Profile</p>', unsafe_allow_html=True)
    col3, col4, col5, col6, col7 = st.columns(5)
    with col3:
        income_options = ["High", "Medium", "Low"]
        income_idx = income_options.index(st.session_state.income_level) if st.session_state.income_level in income_options else 1
        income_level = st.radio(
            "Income Level",
            income_options,
            index=income_idx,
            key="income_level",
            horizontal=False,
            help="Customer income category as provided in the dataset. Exact income range is not specified."
        )
    with col4:
        rating_options = ["High", "Medium", "Low"]
        rating_idx = rating_options.index(st.session_state.credit_rating) if st.session_state.credit_rating in rating_options else 1
        credit_rating = st.radio(
            "Credit Rating",
            rating_options,
            index=rating_idx,
            key="credit_rating",
            horizontal=False,
            help="Customer credit rating category as provided in the dataset."
        )
    with col5:
        household_size = st.number_input(
            "Household Size",
            min_value=1,
            max_value=20,
            step=1,
            key="household_size",
            help="Number of people in the customer's household."
        )
    with col6:
        own_home_options = ["No", "Yes"]
        own_home_idx = own_home_options.index(st.session_state.own_home) if st.session_state.own_home in own_home_options else 0
        own_home = st.radio(
            "Own Your Home",
            own_home_options,
            index=own_home_idx,
            key="own_home",
            horizontal=True,
            help="Whether the customer owns their home."
        )
    with col7:
        # Dynamic Homes Owned behavior logic
        is_disabled = (st.session_state.own_home == "No")
        if is_disabled:
            st.session_state.homes_owned = 0
        else:
            if st.session_state.homes_owned < 1:
                st.session_state.homes_owned = 1

        homes_owned = st.number_input(
            "# Homes Owned",
            min_value=0 if is_disabled else 1,
            max_value=10,
            step=1,
            key="homes_owned",
            disabled=is_disabled,
            help="Number of homes owned by the customer."
        )

    st.markdown("---")

    # Group 3 — Financial Behaviour
    st.markdown('<p class="section-label">Group 3 — Financial Behaviour Details</p>', unsafe_allow_html=True)
    col8, col9, col10 = st.columns(3)
    with col8:
        bank_accounts = st.number_input(
            "# Bank Accounts Open",
            min_value=0,
            max_value=20,
            step=1,
            key="bank_accounts",
            help="Number of bank accounts currently open."
        )
    with col9:
        credit_cards = st.number_input(
            "# Credit Cards Held",
            min_value=0,
            max_value=20,
            step=1,
            key="credit_cards",
            help="Number of credit cards currently held."
        )
    with col10:
        overdraft_options = ["No", "Yes"]
        overdraft_idx = overdraft_options.index(st.session_state.overdraft) if st.session_state.overdraft in overdraft_options else 0
        overdraft = st.radio(
            "Overdraft Protection",
            overdraft_options,
            index=overdraft_idx,
            key="overdraft",
            horizontal=True,
            help="Whether the customer has overdraft protection."
        )

    col11, col12, col13, col14, col15 = st.columns(5)
    with col11:
        avg_placeholder = st.empty()
    with col12:
        q1_balance = st.number_input(
            "Q1 Balance ($)",
            min_value=0.0,
            value=float(st.session_state.get("q1_balance", 500.0)),
            step=50.0,
            format="%.2f",
            help="Customer balance recorded for quarter 1."
        )
        st.session_state["q1_balance"] = q1_balance
    with col13:
        q2_balance = st.number_input(
            "Q2 Balance ($)",
            min_value=0.0,
            value=float(st.session_state.get("q2_balance", 500.0)),
            step=50.0,
            format="%.2f",
            help="Customer balance recorded for quarter 2."
        )
        st.session_state["q2_balance"] = q2_balance
    with col14:
        q3_balance = st.number_input(
            "Q3 Balance ($)",
            min_value=0.0,
            value=float(st.session_state.get("q3_balance", 500.0)),
            step=50.0,
            format="%.2f",
            help="Customer balance recorded for quarter 3."
        )
        st.session_state["q3_balance"] = q3_balance
    with col15:
        q4_balance = st.number_input(
            "Q4 Balance ($)",
            min_value=0.0,
            value=float(st.session_state.get("q4_balance", 500.0)),
            step=50.0,
            format="%.2f",
            help="Customer balance recorded for quarter 4."
        )
        st.session_state["q4_balance"] = q4_balance

    # Auto-calculate Average Balance from the freshly returned quarters
    q_vals = [q1_balance, q2_balance, q3_balance, q4_balance]
    active_qs = [q for q in q_vals if q > 0]
    calc_avg = sum(active_qs) / len(active_qs) if active_qs else 0.0
    st.session_state["avg_balance"] = float(calc_avg)

    with avg_placeholder:
        st.number_input(
            "Average Balance ($)",
            value=float(calc_avg),
            format="%.2f",
            disabled=True,
            help="Automatically calculated from the entered non-zero quarterly balances."
        )

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        predict_btn = st.button("🔍 Predict Acceptance Likelihood", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("🔄 Reset Form", use_container_width=True, on_click=reset_form)

    # ── Prediction logic ────────────────────────
    if predict_btn:
        # Build raw input dict (original column names before get_dummies)
        input_dict = {
            "# Bank Accounts Open": st.session_state.bank_accounts,
            "# Credit Cards Held":  st.session_state.credit_cards,
            "# Homes Owned":        st.session_state.homes_owned,
            "Household Size":       st.session_state.household_size,
            "Average Balance":      st.session_state.avg_balance,
            "Q1 Balance":           st.session_state.q1_balance,
            "Q2 Balance":           st.session_state.q2_balance,
            "Q3 Balance":           st.session_state.q3_balance,
            "Q4 Balance":           st.session_state.q4_balance,
            "Reward":               st.session_state.reward,
            "Mailer Type":          st.session_state.mailer_type,
            "Income Level":         st.session_state.income_level,
            "Overdraft Protection": st.session_state.overdraft,
            "Credit Rating":        st.session_state.credit_rating,
            "Own Your Home":        st.session_state.own_home,
        }

        input_df      = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled  = scaler.transform(input_aligned)

        prob       = model.predict_proba(input_scaled)[0][1]
        prob_pct   = prob * 100
        accepted   = prob >= best_threshold

        # Determine priority
        if prob_pct >= 70:
            priority       = "High Priority"
            rec_text       = "Strong candidate for targeted offer. Prioritise this customer for the campaign."
            box_class      = "result-high"
            badge_emoji    = "🔵"
            decision_label = "Recommended for Targeting"
        elif prob_pct >= 40:
            priority       = "Medium Priority"
            rec_text       = "Consider with campaign budget. Customer may be considered if the campaign budget allows."
            box_class      = "result-medium"
            badge_emoji    = "🟡"
            decision_label = "Consider for Targeting"
        else:
            priority       = "Low Priority"
            rec_text       = "Deprioritise to reduce cost. Lower chance of acceptance."
            box_class      = "result-low"
            badge_emoji    = "⚪"
            decision_label = "Not Recommended for Targeting"

        st.session_state["prediction_result"] = {
            "prob":           prob,
            "prob_pct":       prob_pct,
            "decision_label": decision_label,
            "box_class":      box_class,
            "badge_emoji":    badge_emoji,
            "priority":       priority,
            "rec_text":       rec_text,
        }

        # Append to session live analytics log
        st.session_state["session_predictions"].append({
            "prob_pct": prob_pct,
            "priority": priority,
            "decision": decision_label,
        })

    st.markdown(" ")
    # Model configuration info box below form
    st.info("This prediction uses the final selected model: Threshold-tuned Logistic Regression, with a 70% decision threshold.")

    # ── Display prediction results ──────────────
    if "prediction_result" in st.session_state:
        res = st.session_state["prediction_result"]
        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric(
                "Acceptance Probability",
                f"{res['prob_pct']:.1f}%",
                help="The probability represents the model's estimated likelihood that the customer will accept the credit card offer."
            )
        with r2:
            st.metric(
                "Decision Threshold",
                f"{best_threshold:.0%}",
                help="The minimum probability required for the model to classify a customer as likely to accept. In this dashboard, customers are only predicted as likely to accept if their acceptance probability is 70% or higher."
            )

        with r3:
            st.markdown(
                """
                <div style="font-size: 1rem; font-weight: 600; margin-bottom: 0.35rem;">
                    Marketing Recommendation
                    <span title="Marketing action suggested based on the predicted acceptance probability and the selected decision threshold."
                        style="color: #9ca3af; cursor: help;"> ⓘ</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="
                    font-size: 2.25rem;
                    font-weight: 400;
                    line-height: 1.15;
                    color: #31333F;
                    overflow-wrap: break-word;
                    word-break: normal;
                    max-width: 100%;
                ">
                    {res['decision_label']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(min(res['prob'], 1.0))

        st.markdown(
            f"""
            <div class="{res['box_class']}">
                <div class="result-title">{res['badge_emoji']} {res['priority']}</div>
                <div class="result-sub">{res['rec_text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
