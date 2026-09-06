~# Customer Segmentation & Churn Prediction

An end-to-end Machine Learning project that combines **unsupervised** and **supervised** learning to help a business identify which customers are likely to churn — and which segment they belong to — so retention actions can be prioritized effectively.

---

## Project Status

- [x] Data Cleaning & Preprocessing
- [x] Customer Segmentation (PCA + K-Means)
- [x] Churn Prediction (Classification)
- [x] Combined Risk Labeling
- [x] Interactive Dashboard & Explainability (SHAP)
- [x] Deployment (Streamlit Community Cloud)
- [x] Agentic Layer — Churn Analyst Agent (Gemini + Tool Calling)

**Live App**: [customer-churn-segmentation-byaditya.streamlit.app](https://customer-churn-segmentation-byaditya.streamlit.app/)

---

## Problem Statement

Acquiring a new customer costs more than retaining an existing one. This project helps identify:
1. **Which segment does a customer belong to?** (behavior-based grouping)
2. **Will they churn?** (Yes/No, with probability)
3. **What's their overall risk profile?** (a combined, human-readable label — e.g. "Loyal High-Value — Low Risk", "New Customer — High Risk", with a Medium Risk tier for borderline cases)

---

## Dataset

**Telco Customer Churn** (IBM sample dataset, via Kaggle) — 7,043 customers, 21 columns covering demographics, account info, services used, and churn status.

---

## Repository Structure

```
data_cleaning.ipynb     -> Load raw data, clean (missing values, encoding), save cleaned_telco_data.csv
segmentation.ipynb      -> PCA + K-Means clustering, segment analysis
classification.ipynb    -> Train-test split, XGBoost churn model (tuned), feature importance, saved model
app.py                  -> Streamlit dashboard app: segmentation + classification + SHAP explainability + Churn Analyst Agent
picklefiles/            -> Saved model/encoder objects (onehot_encoder, scaler, pca, kmeans, churn_model)
.env                    -> Local-only file holding GOOGLE_API_KEY (not committed; see Setup below)
```

**Run order**: `data_cleaning.ipynb` -> `segmentation.ipynb` -> `classification.ipynb`

**Local setup for the Agentic layer**: create a `.env` file in the project root with:
```
GOOGLE_API_KEY=your-gemini-api-key
```
(On Streamlit Community Cloud, this is instead set via the app's Secrets settings.)

---

## 1. Data Cleaning

- Dropped `customerID` (no predictive value)
- Fixed `TotalCharges` — was stored as text with hidden blank values; converted to numeric, ~11 rows with missing values dropped (0.15% of data, churn distribution unaffected)
- Encoded target (`Churn`) and binary columns (gender, Partner, Dependents, PhoneService, PaperlessBilling) via Label Encoding
- One-Hot Encoded multi-category columns (Contract, PaymentMethod, InternetService, and service-related columns) with `drop_first=True` to avoid redundancy
- Final cleaned shape: (7032, 31), fully numeric

---

## 2. Customer Segmentation

**Approach**: Selected 26 behavioral/engagement features (tenure, charges, contract, payment method, services used) — excluded demographic columns (gender, SeniorCitizen, Partner, Dependents) and the target variable (Churn), since segmentation should reflect behavior, not outcome or demographics.

- Scaled features with `StandardScaler` (required before PCA, since features had very different ranges)
- Applied **PCA**, reducing 26 dimensions to **11 components** (~90% variance retained)
- Determined optimal cluster count via **Elbow Method** (WCSS) -> suggested k=4
- Cross-validated with **Silhouette Score** -> k=2 scored highest (0.42), but k=4 scored reasonably well (0.32) and was chosen for producing more business-actionable segments
- Visualized clusters in 2D and 3D (using top principal components)

### Segments Identified

| Segment | Description | Tenure | Monthly Charges | Churn Rate | Size |
|---|---|---|---|---|---|
| 0 | Loyal High-Value | 56 mo (highest) | Rs.92 (highest) | ~15% (low) | 29% |
| 1 | Budget, Phone-Only | 31 mo | Rs.21 (lowest) | ~7% (lowest) | 22% |
| 2 | New Customer | 16 mo (lowest) | Rs.74 | ~46% (highest) | 40% (largest) |
| 3 | Internet-Only, No Phone | 32 mo | Rs.42 | ~25% | 10% |

**Key insight**: Segment 2 is both the largest and highest-risk group — retention efforts should prioritize this segment.

---

## 3. Churn Prediction

**Approach**: Used the original (non-PCA) features so feature importance stays interpretable for the business.

- Compared three models: Random Forest, Gradient Boosting, and XGBoost
  - Random Forest overfit heavily (near-perfect train scores, weak test scores)
  - Gradient Boosting generalized well but had lower recall
  - **XGBoost** gave the best balance and was selected for tuning
- Tuned XGBoost via **RandomizedSearchCV** (50 candidates, 5-fold CV) across `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`, and other regularization parameters
- Scoring metric set to **F1** (balances precision and recall) rather than recall alone, since maximizing recall alone pushed precision too low (excessive false alarms)
- Best `scale_pos_weight` = 2.76, matching the actual class imbalance ratio in the training data

**Final Test Performance**:
- Recall: ~0.73 (majority of actual churners correctly identified)
- Precision: ~0.55
- Minimal overfitting (train-test recall gap ~0.02)

**Feature Importance & Simplification**: Checked `feature_importances_` and found demographic columns (gender, SeniorCitizen, Partner, Dependents) contributed minimally (~2.7% combined), and three "No internet service" dummy columns had exactly zero importance (redundant with `InternetService_No`). Dropped all 8 columns — performance was unaffected, and the deployment UI will need fewer input fields.

Top churn drivers: `InternetService_Fiber optic`, `Contract_One year`, and internet-service-related features (OnlineSecurity, TechSupport, OnlineBackup) — consistent with the segmentation finding that Segment 2 (New Customer) skews toward month-to-month contracts.

Final model saved as `churn_model.pkl`.

---

## 4. Combined Risk Labeling

For a single customer's input, the app computes both outputs together and combines them into one readable label:
- Segment (from the segmentation pipeline: scaler -> PCA -> K-Means) gives the customer's behavioral group (e.g. "Loyal High-Value", "New Customer")
- Churn probability (from the tuned XGBoost model) is classified into a **3-tier risk level**:
  - **Low Risk**: probability < 35%
  - **Medium Risk**: 35% – 60%
  - **High Risk**: probability >= 60%
- A hard 50% cutoff was tried first, but borderline predictions (e.g. 47%) were being labeled "Low Risk" despite being close to a coin-flip — the 3-tier system avoids this misleading edge case.

This lets a non-technical user (e.g. a retention team member) get an actionable, human-readable output instead of raw numbers.

---

## 5. Interactive Dashboard & Explainability

The app (`app.py`) is a **Streamlit** single-customer predictor with a full dashboard-style output, built to be presentable for a portfolio/viva, not just a bare prediction:

- **Risk card**: color-coded (green/yellow/red) result with the predicted segment
- **SHAP explanation**: per-customer reasoning using `shap.TreeExplainer` on the XGBoost model — shows exactly which of *this customer's* attributes pushed their churn probability up (red) or down (blue), as opposed to generic dataset-wide importance
- **Segment reasoning**: compares the customer's tenure/monthly charges against their segment's averages, with a plain-language description of what that segment typically looks like
- **Churn probability gauge**: speedometer-style chart with color zones matching the 3-tier thresholds
- **Customer base by segment**: pie chart showing where this customer sits relative to the overall segment distribution
- **Tenure/Charges comparison bars**: this customer vs. their segment's average
- **Top churn drivers**: global feature importance from the trained model, shown as a horizontal bar chart

Each section is rendered in its own bordered card (`st.container(border=True)`) with a consistent labeled header, for a clean dashboard layout rather than a plain scrolling form.

---

## 6. Agentic Layer — Churn Analyst Agent

Beyond static prediction and explanation, the app includes an **AI agent** that reasons over a prediction and decides what to do next, rather than just describing it. This is built with the **Gemini API's tool-calling (function-calling) feature** — Gemini is given the customer's segment, risk level, churn probability, and top SHAP factors, along with a set of Python functions ("tools") it can choose to call.

**The flow**: Predict → Interpret → Decide → Act
1. **Interpret**: Gemini explains, in plain language, *why* this customer has this risk level, grounded in the SHAP factors already computed.
2. **Decide**: Gemini calls `check_retention_priority(risk_level, monthly_charges)` — a deterministic business-rule function (not something the model should guess on its own) that returns a priority level (e.g. "URGENT", "MEDIUM PRIORITY", "NO ACTION NEEDED").
3. **Act**: If the priority indicates action is needed, Gemini calls `draft_retention_email(segment, priority, key_reasons)`, which fills a template with the customer's context to produce a ready-to-review retention email.

Gemini decides *whether* and *in what order* to call these tools — the sequence is not hardcoded with `if/else`. The important distinction: Gemini never executes the functions itself; it only requests a call, and the `google-genai` SDK's automatic function calling executes the actual Python function locally and returns the result back to the model to continue reasoning.

This turns the tool from a pure prediction dashboard into a lightweight decision-support system.

---

## Tech Stack

| Component | Tool |
|---|---|
| Data handling | Pandas, NumPy |
| Dimensionality reduction | PCA (scikit-learn) |
| Clustering | K-Means (scikit-learn) |
| Classification | XGBoost, tuned with RandomizedSearchCV |
| Explainability | SHAP (TreeExplainer) |
| Agentic Layer | Google Gemini API (`google-genai`), tool/function calling |
| Config | `python-dotenv` (local `.env` for API keys) |
| Visualization | Matplotlib, Plotly |
| App Interface | Streamlit |
| Deployment | Streamlit Community Cloud |

