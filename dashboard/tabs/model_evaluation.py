import streamlit as st

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def render_model_evaluation(df_results, has_plotly):
    st.title("📊 Model Evaluation")
    st.markdown(
        "This tab presents the model evaluation process used to select the final prediction model. "
        "Since accepted customers represent a small proportion of the dataset, the evaluation focuses on minority-class performance instead of accuracy alone."
    )

    # Final model highlight
    st.markdown('<p class="section-header">Final Selected Model</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.info(
            "**Threshold-tuned Logistic Regression** was selected as the final model because "
            "it achieved the strongest overall balance between ROC-AUC, minority-class F1-score, recall, "
            "and interpretability for marketing decision support."
        )

    # Final model metrics
    st.markdown('<p class="section-header">Final Model Metrics</p>', unsafe_allow_html=True)
    with st.container(border=True):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC",   "0.764")
        m2.metric("Accuracy",  "0.86")
        m3.metric("Precision", "0.18")
        m4.metric("Recall",    "0.42")
        m5.metric("F1-Score",  "0.26")

    st.markdown(" ")

    # Model comparison table
    st.markdown('<p class="section-header">Model Comparison Table</p>', unsafe_allow_html=True)
    st.caption("All experiments across Logistic Regression, Random Forest, XGBoost, and LightGBM.")

    if df_results.empty:
        st.warning("Model comparison data is not yet available.")
    else:
        # Highlight the selected model row
        def highlight_selected(row):
            is_selected = (
                str(row.get("Model", "")).strip() == "Logistic Regression"
                and str(row.get("Experiment", "")).strip().lower() == "threshold tuned"
            )
            return ["background-color: #14532d; color: #f0fdf4" if is_selected else "" for _ in row]

        styled = df_results.style.apply(highlight_selected, axis=1)
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model":          st.column_config.TextColumn("Model"),
                "Experiment":     st.column_config.TextColumn("Experiment"),
                "Accuracy":       st.column_config.NumberColumn("Accuracy",  format="%.2f"),
                "Precision":      st.column_config.NumberColumn("Precision", format="%.2f"),
                "Recall":         st.column_config.NumberColumn("Recall",    format="%.2f"),
                "F1-Score":       st.column_config.NumberColumn("F1-Score",  format="%.2f"),
                "ROC-AUC":        st.column_config.NumberColumn("ROC-AUC",   format="%.2f"),
                "Confusion Matrix": None,   # Hidden — keeps focus on core metrics
                "Interpretation": st.column_config.TextColumn("Key Takeaway / Interpretation"),
            }
        )
        st.caption("🟢 Highlighted row = final selected model")

    st.divider()

    # Why this model — two-column layout
    st.markdown('<p class="section-header">Why Logistic Regression?</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col_why1, col_why2 = st.columns(2)
        with col_why1:
            st.markdown("""
            - **Highest ROC-AUC (0.764):** Demonstrates the strongest overall discrimination ability between accepting and non-accepting customers.
            - **Threshold Tuning Benefit:** Applying a 0.70 threshold significantly improved precision and F1-score for the minority class compared to the default 0.50 threshold.
            - **Transparent Interpretability:** Logistic Regression provides clear, mathematically explainable coefficients suited to a professional decision-support system.
            """)
        with col_why2:
            st.markdown("""
            - **Campaign Feature Benefit:** The full-feature model outperformed the customer-only variant, proving that campaign-specific features (reward type, mailer type) are powerful predictors.
            - **Stronger than Alternatives:** Random Forest produced zero recall on the minority class; XGBoost and LightGBM showed lower ROC-AUC and F1-scores across all experiments.
            """)

    # ── Model comparison bar charts ───────────────
    if not df_results.empty and has_plotly:
        st.markdown('<p class="section-header">Model Performance Charts</p>', unsafe_allow_html=True)
        st.caption("These charts compare model performance across experiments, with emphasis on minority-class detection.")

        df_plot = df_results.copy()
        df_plot["Model & Experiment"] = df_plot["Model"] + " — " + df_plot["Experiment"]
        df_plot["Is Selected"]    = (df_plot["Model"] == "Logistic Regression") & (df_plot["Experiment"].str.lower() == "threshold tuned")
        df_plot["Selected Label"] = df_plot["Is Selected"].map({True: "Final Selected Model", False: "Alternative Model"})

        # ROC-AUC Expander (expanded by default)
        with st.expander("ROC-AUC Comparison", expanded=True):
            st.subheader(
                "ROC-AUC Comparison",
                help="ROC-AUC measures how well the model separates accepted and non-accepted customers. Higher is better."
            )
            fig_roc = px.bar(
                df_plot,
                x="ROC-AUC",
                y="Model & Experiment",
                orientation="h",
                color="Model",
                hover_data={"Model": True, "Experiment": True, "Selected Label": True, "ROC-AUC": ":.3f"},
                category_orders={"Model & Experiment": df_plot.sort_values("ROC-AUC")["Model & Experiment"].tolist()}
            )
            fig_roc.add_vline(
                x=0.764,
                line_dash="dash",
                line_color="green",
                annotation_text="Selected: 0.764",
                annotation_position="top right"
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # F1-Score Expander (collapsed by default)
        with st.expander("F1-Score Comparison"):
            st.subheader(
                "F1-Score Comparison",
                help="F1-Score balances precision and recall for the minority (accepted) class. Higher is better."
            )
            fig_f1 = px.bar(
                df_plot,
                x="F1-Score",
                y="Model & Experiment",
                orientation="h",
                color="Model",
                hover_data={"Model": True, "Experiment": True, "Selected Label": True, "F1-Score": ":.3f"},
                category_orders={"Model & Experiment": df_plot.sort_values("F1-Score")["Model & Experiment"].tolist()}
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        # Recall Expander (collapsed by default)
        with st.expander("Recall Comparison"):
            st.subheader(
                "Recall Comparison",
                help="Recall measures how many actual acceptors the model correctly identifies. Higher recall reduces missed campaign targets. Higher is better."
            )
            fig_recall = px.bar(
                df_plot,
                x="Recall",
                y="Model & Experiment",
                orientation="h",
                color="Model",
                hover_data={"Model": True, "Experiment": True, "Selected Label": True, "Recall": ":.3f"},
                category_orders={"Model & Experiment": df_plot.sort_values("Recall")["Model & Experiment"].tolist()}
            )
            st.plotly_chart(fig_recall, use_container_width=True)
