import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def render_data_insights(df_hist, has_plotly):
    st.title("📈 Data Insights")
    st.info(
        "This tab explores patterns in the historical dataset used for model development. "
        "These charts help explain customer and campaign characteristics related to offer acceptance. "
        "They are descriptive insights only and do not represent live deployment data."
    )
    st.divider()

    if df_hist is not None and not df_hist.empty:
        # Sort categoricals beforehand
        df_hist["Income Level"] = pd.Categorical(df_hist["Income Level"], categories=["Low", "Medium", "High"], ordered=True)
        df_hist["Credit Rating"] = pd.Categorical(df_hist["Credit Rating"], categories=["Low", "Medium", "High"], ordered=True)

        # Dataset Snapshot Metrics
        total_records  = len(df_hist)
        accepted_count = int(df_hist["target"].sum())
        acceptance_rate = (accepted_count / total_records) * 100 if total_records > 0 else 0.0

        st.markdown('<p class="section-header">Dataset Snapshot</p>', unsafe_allow_html=True)
        snap_col1, snap_col2, snap_col3 = st.columns(3)
        with snap_col1:
            st.metric("Total Records", f"{total_records:,}")
        with snap_col2:
            st.metric("Accepted Offers", f"{accepted_count:,}")
        with snap_col3:
            st.metric("Historical Acceptance Rate", f"{acceptance_rate:.2f}%")
        st.divider()

        # Row 1
        di1, di2 = st.columns(2)

        with di1:
            with st.container(border=True):
                st.markdown('<div class="info-card" style="margin-bottom: 0;"><h4>Offer Acceptance Distribution</h4><p style="font-weight: 400; font-size: 0.9rem; color: #6b7280; margin-top: 0.2rem;">The dataset is highly imbalanced — only ~5.68% of customers accepted the offer.</p></div>', unsafe_allow_html=True)
                dist = df_hist["target"].value_counts().reset_index()
                dist.columns = ["Status", "Count"]
                dist["Status"] = dist["Status"].map({0: "Not Accepted", 1: "Accepted"})
                if has_plotly:
                    fig = px.pie(
                        dist,
                        values="Count",
                        names="Status",
                        color="Status",
                        color_discrete_map={"Accepted": "#059669", "Not Accepted": "#64748b"},
                        hole=0.45,
                        title="Offer Acceptance Ratio"
                    )
                    fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(dist.set_index("Status"))

        with di2:
            with st.container(border=True):
                st.markdown('<div class="info-card" style="margin-bottom: 0;"><h4>Acceptance Rate by Reward</h4><p style="font-weight: 400; font-size: 0.9rem; color: #6b7280; margin-top: 0.2rem;">Reward type may influence customer willingness to accept an offer.</p></div>', unsafe_allow_html=True)
                reward_acc = df_hist.groupby("Reward")["target"].mean().reset_index()
                reward_acc.columns = ["Reward", "Acceptance Rate"]
                reward_acc["Acceptance Rate %"] = reward_acc["Acceptance Rate"] * 100
                if has_plotly:
                    fig = px.bar(
                        reward_acc, x="Reward", y="Acceptance Rate %", color="Reward",
                        title="Acceptance Rate by Reward Type"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(reward_acc.set_index("Reward")["Acceptance Rate %"])

        # Row 2
        di3, di4 = st.columns(2)

        with di3:
            with st.container(border=True):
                st.markdown('<div class="info-card" style="margin-bottom: 0;"><h4>Acceptance Rate by Mailer Type</h4><p style="font-weight: 400; font-size: 0.9rem; color: #6b7280; margin-top: 0.2rem;">Postcard vs Letter mailers may have different response rates among customers.</p></div>', unsafe_allow_html=True)
                mailer_acc = df_hist.groupby("Mailer Type")["target"].mean().reset_index()
                mailer_acc.columns = ["Mailer Type", "Acceptance Rate"]
                mailer_acc["Acceptance Rate %"] = mailer_acc["Acceptance Rate"] * 100
                if has_plotly:
                    fig = px.bar(
                        mailer_acc, x="Mailer Type", y="Acceptance Rate %", color="Mailer Type",
                        title="Acceptance Rate by Mailer Type"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(mailer_acc.set_index("Mailer Type")["Acceptance Rate %"])

        with di4:
            with st.container(border=True):
                st.markdown('<div class="info-card" style="margin-bottom: 0;"><h4>Acceptance Rate by Income Level</h4><p style="font-weight: 400; font-size: 0.9rem; color: #6b7280; margin-top: 0.2rem;">Customer income level may be a useful feature for targeting campaigns.</p></div>', unsafe_allow_html=True)
                inc_acc = df_hist.groupby("Income Level", observed=False)["target"].mean().reset_index()
                inc_acc.columns = ["Income Level", "Acceptance Rate"]
                inc_acc["Acceptance Rate %"] = inc_acc["Acceptance Rate"] * 100
                if has_plotly:
                    fig = px.bar(
                        inc_acc, x="Income Level", y="Acceptance Rate %", color="Income Level",
                        title="Acceptance Rate by Income Level"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(inc_acc.set_index("Income Level")["Acceptance Rate %"])

        # Row 3
        di5, di6 = st.columns(2)

        with di5:
            with st.container(border=True):
                st.markdown('<div class="info-card" style="margin-bottom: 0;"><h4>Acceptance Rate by Credit Rating</h4><p style="font-weight: 400; font-size: 0.9rem; color: #6b7280; margin-top: 0.2rem;">Credit rating may reflect customer financial confidence and willingness to take on new products.</p></div>', unsafe_allow_html=True)
                cr_acc = df_hist.groupby("Credit Rating", observed=False)["target"].mean().reset_index()
                cr_acc.columns = ["Credit Rating", "Acceptance Rate"]
                cr_acc["Acceptance Rate %"] = cr_acc["Acceptance Rate"] * 100
                if has_plotly:
                    fig = px.bar(
                        cr_acc, x="Credit Rating", y="Acceptance Rate %", color="Credit Rating",
                        title="Acceptance Rate by Credit Rating"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(cr_acc.set_index("Credit Rating")["Acceptance Rate %"])

        with di6:
            with st.container(border=True):
                st.markdown('<div class="info-card" style="margin-bottom: 0;"><h4>Average Balance by Offer Acceptance</h4><p style="font-weight: 400; font-size: 0.9rem; color: #6b7280; margin-top: 0.2rem;">Customers who accepted the offer may show different average balance patterns.</p></div>', unsafe_allow_html=True)
                df_hist["Status"] = df_hist["target"].map({0: "Not Accepted", 1: "Accepted"})
                if has_plotly:
                    fig = px.box(
                        df_hist,
                        x="Status",
                        y="Average Balance",
                        color="Status",
                        color_discrete_map={"Accepted": "#059669", "Not Accepted": "#64748b"},
                        title="Average Balance Distribution by Offer Acceptance Status"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    bal_acc = df_hist.groupby("target")["Average Balance"].mean().reset_index()
                    bal_acc["Status"] = bal_acc["target"].map({0: "Not Accepted", 1: "Accepted"})
                    st.bar_chart(bal_acc.set_index("Status")["Average Balance"])

        # Row 4 — Line chart and Key Insight box
        di7, di8 = st.columns(2)

        with di7:
            with st.container(border=True):
                st.markdown('<div class="info-card" style="margin-bottom: 0;"><h4>Quarterly Balance Trend</h4><p style="font-weight: 400; font-size: 0.9rem; color: #6b7280; margin-top: 0.2rem;">Quarterly balance trend highlights how average customer balances change over the four quarters.</p></div>', unsafe_allow_html=True)
                q_cols = ["Q1 Balance", "Q2 Balance", "Q3 Balance", "Q4 Balance"]
                q_balances = df_hist.groupby("target")[q_cols].mean().reset_index()
                q_balances_melted = q_balances.melt(
                    id_vars="target",
                    value_vars=q_cols,
                    var_name="Quarter",
                    value_name="Average Balance"
                )
                q_balances_melted["Status"] = q_balances_melted["target"].map({0: "Not Accepted", 1: "Accepted"})
                if has_plotly:
                    fig_line = px.line(
                        q_balances_melted,
                        x="Quarter",
                        y="Average Balance",
                        color="Status",
                        color_discrete_map={"Accepted": "#059669", "Not Accepted": "#64748b"},
                        markers=True,
                        title="Average Quarterly Balance Trend by Acceptance Status"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.line_chart(q_balances_melted.pivot(index="Quarter", columns="Status", values="Average Balance"))

        with di8:
            with st.container(border=True):
                st.markdown('<div class="info-card" style="margin-bottom: 0;"><h4>Dataset Imbalance Analysis</h4></div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.warning(
                    "💡 **Key Insight:** The dataset is highly imbalanced, with a small proportion of accepted offers. "
                    "This is why model evaluation should consider minority-class performance and not rely on accuracy alone."
                )
    else:
        st.warning("Historical dataset could not be loaded.")
