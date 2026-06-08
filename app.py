import sys
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

try:
    import plotly.express as px  # noqa: F401 — kept for HAS_PLOTLY check
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ==================================================
# Paths  (app.py lives at repo root)
# ==================================================
BASE_DIR   = Path(__file__).resolve().parent          # repo root
DASH_DIR   = BASE_DIR / "dashboard"
MODELS_DIR = BASE_DIR / "models" / "saved_models"

# Allow  `from tabs.xxx import ...`  to resolve against dashboard/
if str(DASH_DIR) not in sys.path:
    sys.path.insert(0, str(DASH_DIR))

# ==================================================
# Page config
# ==================================================
st.set_page_config(
    page_title="Credit Card Offer Acceptance Dashboard",
    page_icon="💳",
    layout="wide",
)

# ==================================================
# CSS Loader
# ==================================================
def load_css(file_path: Path) -> None:
    """Load an external CSS file and inject it into the Streamlit app."""
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_path}")

load_css(DASH_DIR / "styles.css")

# ==================================================
# Tab imports  (resolved via sys.path above)
# ==================================================
from tabs.overview         import render_overview          # noqa: E402
from tabs.prediction       import render_prediction        # noqa: E402
from tabs.data_insights    import render_data_insights     # noqa: E402
from tabs.model_evaluation import render_model_evaluation  # noqa: E402
from tabs.about            import render_about             # noqa: E402

# ==================================================
# Load model assets (cached)
# ==================================================
@st.cache_resource
def load_model_assets():
    model     = joblib.load(MODELS_DIR / "final_logistic_regression_model.pkl")
    scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
    columns   = joblib.load(MODELS_DIR / "model_columns.pkl")
    threshold = joblib.load(MODELS_DIR / "best_threshold.pkl")
    return model, scaler, columns, threshold

@st.cache_data
def load_model_results():
    return pd.read_csv(DASH_DIR / "model_results.csv")

@st.cache_data
def load_dataset():
    return pd.read_csv(BASE_DIR / "data" / "processed" / "final_dataset.csv")

model, scaler, model_columns, best_threshold = load_model_assets()
df_results = load_model_results()
df_hist    = load_dataset()

# ==================================================
# Default Input Values & Session State Setup
# ==================================================
DEFAULT_FORM_VALUES = {
    "reward":         "Air Miles",
    "mailer_type":    "Letter",
    "income_level":   "Medium",
    "credit_rating":  "Medium",
    "household_size": 2,
    "own_home":       "No",
    "homes_owned":    0,
    "bank_accounts":  1,
    "credit_cards":   1,
    "overdraft":      "No",
    "avg_balance":    500.0,
    "q1_balance":     500.0,
    "q2_balance":     500.0,
    "q3_balance":     500.0,
    "q4_balance":     500.0,
}

def reset_form():
    for k, v in DEFAULT_FORM_VALUES.items():
        st.session_state[k] = v
    if "prediction_result" in st.session_state:
        del st.session_state["prediction_result"]

for k, v in DEFAULT_FORM_VALUES.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "session_predictions" not in st.session_state:
    st.session_state["session_predictions"] = []

# ==================================================
# Stateful Navigation Tabs
# ==================================================
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Overview"

tabs_list = ["Overview", "Prediction", "Data Insights", "Model Evaluation", "About"]

cols = st.columns(len(tabs_list))
for idx, t_name in enumerate(tabs_list):
    with cols[idx]:
        if st.button(
            t_name,
            use_container_width=True,
            type="primary" if st.session_state.active_tab == t_name else "secondary"
        ):
            st.session_state.active_tab = t_name
            st.rerun()

# ==================================================
# Tab Routing
# ==================================================
active = st.session_state.active_tab

if active == "Overview":
    render_overview(df_hist)

elif active == "Prediction":
    render_prediction(model, scaler, model_columns, best_threshold, reset_form)

elif active == "Data Insights":
    render_data_insights(df_hist, HAS_PLOTLY)

elif active == "Model Evaluation":
    render_model_evaluation(df_results, HAS_PLOTLY)

elif active == "About":
    render_about()
