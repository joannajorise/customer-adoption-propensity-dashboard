# Predicting Customer Adoption Propensity Using Spending Behaviour

A machine learning dashboard prototype that predicts whether a bank customer is likely to accept a credit card offer, built as a Final Year Project (FYP).

---


## 🌐 Live Dashboard

Access the deployed dashboard here:

[Customer Adoption Propensity Dashboard](https://customer-adoption-propensity.streamlit.app/)

This interactive dashboard allows users to:

* Predict customer credit card offer acceptance likelihood
* Explore customer and campaign insights
* Compare machine learning model performance
* Support targeted marketing decision-making

---

## 📌 Project Overview

This project demonstrates how machine learning can support **targeted credit card offer marketing** by predicting customer response propensity. A bank marketing officer enters customer profile and campaign details into the dashboard and receives an **offer acceptance probability** along with a **marketing priority recommendation**.

The final selected model is a **Threshold-tuned Logistic Regression** trained on historical customer and campaign data. The dashboard loads the saved model for inference — it does not retrain the model at runtime.

---

## ⚠️ Scope Clarification

This dashboard is **NOT**:
- A credit approval system
- A creditworthiness assessment tool
- A loan eligibility predictor
- A fraud detection system
- A live banking deployment

It is **only** a prototype for **marketing response prediction** — predicting which customers are more likely to accept a credit card offer for targeted campaign support.

---

## 📁 Folder Structure

```
FYP_Project/
├── app.py                          # Root-level Streamlit entry point (for Streamlit Cloud)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git exclusions
│
├── dashboard/                      # Dashboard application
│   ├── app.py                      # Local entry point (run from dashboard/)
│   ├── styles.css                  # Custom UI styling
│   ├── model_results.csv           # Model comparison results for evaluation tab
│   ├── dashboard_design_brief.md   # UI/UX design notes
│   └── tabs/                       # Individual tab modules
│       ├── __init__.py
│       ├── overview.py
│       ├── prediction.py
│       ├── data_insights.py
│       ├── model_evaluation.py
│       └── about.py
│
├── models/
│   ├── saved_models/               # Trained model files (required by dashboard)
│   │   ├── final_logistic_regression_model.pkl
│   │   ├── scaler.pkl
│   │   ├── model_columns.pkl
│   │   └── best_threshold.pkl
│   ├── baseline/                   # Baseline model training scripts
│   ├── experiments/                # Feature experiment scripts
│   └── tuning/                     # Hyperparameter tuning scripts
│
└── data/
    ├── raw/
    │   └── credit_card_offer_acceptance.csv    # Original dataset
    └── processed/
        └── final_dataset.csv                   # Preprocessed dataset (used by dashboard)
```

---

## 📊 Dataset Overview

| Property | Detail |
|---|---|
| **Name** | Credit Card Offer Acceptance Trends in Banking |
| **Records** | 18,000 customers |
| **Target variable** | `target` — whether a customer accepted the credit card offer (1 = Accepted, 0 = Not Accepted) |
| **Accepted offers** | 1,023 (~5.68%) |
| **Class imbalance** | Highly imbalanced — minority class handled with `class_weight='balanced'` |
| **Train/Test split** | 80% training / 20% testing |
| **Features** | Customer profile, banking relationship, financial behaviour, campaign details |

---

## 🤖 Final Selected Model

| Property | Detail |
|---|---|
| **Model** | Logistic Regression |
| **Configuration** | Threshold-tuned (threshold = 0.70), `class_weight='balanced'`, `max_iter=1000` |
| **ROC-AUC** | 0.764 |
| **Recall (minority class)** | 0.42 |
| **F1-Score (minority class)** | 0.26 |
| **Accuracy** | 0.86 |
| **Why selected** | Highest ROC-AUC, strongest minority-class F1 after threshold tuning, interpretable |

---

## 🖥️ Dashboard Features

The dashboard has five tabs:

| Tab | Description |
|---|---|
| **Overview** | Project summary, key metrics, simulated live analytics |
| **Prediction** | Enter customer and campaign inputs to get acceptance probability + marketing recommendation |
| **Data Insights** | Descriptive charts from the historical dataset (acceptance distributions, quarterly trends) |
| **Model Evaluation** | Model comparison table and performance charts (ROC-AUC, F1, Recall) |
| **About** | Dashboard purpose, limitations, intended users, dataset info |

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/joannajorise/customer-adoption-propensity-dashboard.git
cd customer-adoption-propensity-dashboard
```


### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

**From the repo root (recommended):**
```bash
streamlit run dashboard/app.py
```

**Or from the dashboard folder:**
```bash
cd dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deployment

This dashboard is deployed using **Streamlit Community Cloud**.

Deployment configuration:

* **Repository:** `customer-adoption-propensity-dashboard`
* **Branch:** `main`
* **Main file path:** `dashboard/app.py`

The deployment automatically installs dependencies from `requirements.txt`.

---

## ⚙️ Limitations

- **Prototype only:** Not intended for live production banking use.
- **Historical data:** Predictions reflect patterns from the training dataset and may not generalise to all future scenarios.
- **Imbalanced dataset:** The accepted class is a small minority; the model is optimised for recall at the cost of precision.
- **No real-time data:** The dashboard uses manual input and a saved model. It is not connected to live banking databases.
- **Academic context:** Developed as part of a Final Year Project (FYP) research study.

---

## 👩‍💻 Author

**Joanna Jorise**  
82623
Bachelor of Computer Science with Honours (Computational Science) 
Universiti Malaysia Sarawak (UNIMAS)  
82623@siswa.unimas.my

---

## 📝 Disclaimer

This dashboard is developed for academic purposes as part of a Final Year Project. Predictions are based on a prototype machine learning model and should not be used for actual credit decisions.
