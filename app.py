import os
import pandas as pd
import pickle
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #4338CA;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        margin-top: 1rem;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border: 1px solid #F87171;
        color: #7F1D1D;
    }
    .risk-medium {
        background: linear-gradient(135deg, #FEF3C7, #FDE68A);
        border: 1px solid #FBBF24;
        color: #78350F;
    }
    .risk-low {
        background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
        border: 1px solid #4ADE80;
        color: #14532D;
    }
    .risk-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: inherit;
    }
    .result-card div {
        color: inherit;
    }
    .segment-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background-color: #4338CA;
        color: white !important;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4338CA;
    }
    .card-label {
        display: inline-block;
        background-color: #4338CA;
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        letter-spacing: 0.04em;
        margin-bottom: 0.7rem;
        text-transform: uppercase;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 1rem 1.2rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load saved models/encoders (all fit during training)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    churn_model = pickle.load(open('picklefiles/churn_model.pkl', 'rb'))
    ohe = pickle.load(open('picklefiles/onehot_encoder.pkl', 'rb'))
    scaler = pickle.load(open('picklefiles/scale.pkl', 'rb'))
    pca = pickle.load(open('picklefiles/pca.pkl', 'rb'))
    kmeans = pickle.load(open('picklefiles/kmeans.pkl', 'rb'))
    return churn_model, ohe, scaler, pca, kmeans

churn_model, ohe, scaler, pca, kmeans = load_artifacts()

@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(churn_model)

explainer = get_explainer()

# ---------------------------------------------------------
# Column lists — MUST match training exactly (names + order)
# ---------------------------------------------------------
CAT_COLS = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod']

SEGMENTATION_COLS = [
    'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

DROP_FOR_CLASSIFICATION = [
    'DeviceProtection_No internet service',
    'StreamingTV_No internet service',
    'StreamingMovies_No internet service'
]
CLASSIFICATION_COLS = [c for c in SEGMENTATION_COLS if c not in DROP_FOR_CLASSIFICATION]

SEGMENT_NAMES = {
    0: "Loyal High-Value",
    1: "Budget, Phone-Only",
    2: "New Customer",
    3: "Internet-Only, No Phone"
}

# Reference stats from the segmentation analysis (group-level averages, not the model)
SEGMENT_STATS = {
    0: {"share": 29.0, "avg_tenure": 56.4, "avg_monthly": 92.3, "avg_churn": 14.8},
    1: {"share": 21.6, "avg_tenure": 31.0, "avg_monthly": 21.1, "avg_churn": 7.4},
    2: {"share": 39.7, "avg_tenure": 16.1, "avg_monthly": 74.0, "avg_churn": 46.0},
    3: {"share": 9.7, "avg_tenure": 32.0, "avg_monthly": 42.0, "avg_churn": 25.0},
}

SEGMENT_DESCRIPTIONS = {
    0: "long-tenured customers with high monthly spend, mostly on 1-2 year contracts — the most stable, highest-value group.",
    1: "customers with no internet service, low monthly charges, and moderate tenure — a budget, low-engagement group.",
    2: "recently joined customers, mostly on month-to-month contracts with high churn — the group needing the most retention attention.",
    3: "customers with internet service but no phone line, moderate tenure and spend."
}


def get_shap_reasons(classification_input, top_n=5):
    """Real SHAP values for this specific customer, using the model's own TreeExplainer.
    Positive SHAP value = pushes prediction toward churn. Negative = pushes away from churn."""
    shap_values = explainer.shap_values(classification_input)[0]
    base_value = explainer.expected_value
    if hasattr(base_value, "__len__"):
        base_value = base_value[0]

    shap_df = pd.DataFrame({
        'Feature': CLASSIFICATION_COLS,
        'Value': classification_input.iloc[0].values,
        'SHAP Value': shap_values
    })
    shap_df['abs'] = shap_df['SHAP Value'].abs()
    shap_df = shap_df.sort_values('abs', ascending=False).head(top_n).sort_values('SHAP Value')

    return shap_df, base_value


def format_feature_name(name):
    """Turn a one-hot column name into a readable label (no value context)."""
    return name.replace('_', ': ').replace('No internet service', 'No Internet')


def format_feature_label(feature, value):
    """Format a feature for SHAP display, making clear whether the customer actually
    HAS that attribute (dummy=1) or NOT (dummy=0) — a one-hot column at 0 still gets a
    SHAP value (the effect of *not* having that category), which is easy to misread."""
    if feature in ('PhoneService', 'PaperlessBilling'):
        val_str = 'Yes' if value == 1 else 'No'
        return f"{feature}: {val_str}"

    if '_' in feature:
        base, cat = feature.split('_', 1)
        cat = cat.replace('No internet service', 'No Internet').replace('No phone service', 'No Phone')
        if value == 1:
            return f"{base}: {cat}"
        else:
            return f"{base} ≠ {cat}"

    # continuous features: tenure, MonthlyCharges, TotalCharges
    return f"{feature} = {value:.0f}"


def predict_customer(tenure, phone_service, multiple_lines, internet_service,
                      online_security, online_backup, device_protection,
                      tech_support, streaming_tv, streaming_movies,
                      contract, paperless_billing, payment_method,
                      monthly_charges, total_charges):

    raw = pd.DataFrame([{
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    raw['PhoneService'] = raw['PhoneService'].map({'No': 0, 'Yes': 1})
    raw['PaperlessBilling'] = raw['PaperlessBilling'].map({'No': 0, 'Yes': 1})

    encoded_array = ohe.transform(raw[CAT_COLS])
    encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(CAT_COLS))

    final_df = pd.concat(
        [raw.drop(columns=CAT_COLS).reset_index(drop=True), encoded_df],
        axis=1
    )

    segmentation_input = final_df[SEGMENTATION_COLS]
    scaled = scaler.transform(segmentation_input)
    pca_transformed = pca.transform(scaled)
    segment = int(kmeans.predict(pca_transformed)[0])
    segment_name = SEGMENT_NAMES[segment]

    classification_input = final_df[CLASSIFICATION_COLS]
    churn_prob = churn_model.predict_proba(classification_input)[0][1]

    if churn_prob >= 0.6:
        risk = "High Risk"
    elif churn_prob >= 0.35:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"
    return segment_name, risk, churn_prob, classification_input


# ---------------------------------------------------------
# Churn Analyst Agent — Gemini + Tool Calling
# ---------------------------------------------------------
@st.cache_resource
def get_genai_client():
    # Local development reads from .env; Streamlit Cloud reads from st.secrets.
    # st.secrets access raises if no secrets.toml exists at all (normal for local dev),
    # so that lookup is wrapped safely.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY", None)
        except Exception:
            api_key = None
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

genai_client = get_genai_client()


def check_retention_priority(risk_level: str, monthly_charges: float) -> str:
    """Look up this company's retention-priority policy for a customer, based on
    their churn risk level and how much they pay per month. This encodes business
    rules that are not something a language model should guess on its own.

    Args:
        risk_level: One of "Low Risk", "Medium Risk", "High Risk".
        monthly_charges: The customer's monthly charges in rupees.

    Returns:
        A short priority label and recommended action timeline.
    """
    if risk_level == "High Risk" and monthly_charges >= 70:
        return "URGENT: high-value customer at high churn risk. Recommend personalized outreach within 48 hours."
    elif risk_level == "High Risk":
        return "HIGH PRIORITY: schedule retention contact within 1 week."
    elif risk_level == "Medium Risk" and monthly_charges >= 70:
        return "MEDIUM PRIORITY: high-value customer showing risk signs. Recommend proactive check-in."
    elif risk_level == "Medium Risk":
        return "MEDIUM PRIORITY: include in the next standard retention campaign."
    else:
        return "NO ACTION NEEDED: customer is low risk."


def draft_retention_email(segment: str, priority: str, key_reasons: str) -> str:
    """Generate a retention email draft using the company's standard template,
    personalized with the customer's segment and the key factors driving their
    churn risk. Only call this if check_retention_priority indicated that some
    action is needed (i.e. priority is not "NO ACTION NEEDED").

    Args:
        segment: The customer's behavioral segment name.
        priority: The priority label returned by check_retention_priority.
        key_reasons: A short, comma-separated description of the top factors
            driving this customer's churn risk (e.g. "month-to-month contract,
            fiber optic internet, electronic check payment").

    Returns:
        A draft email with a subject line and body, ready for a retention agent
        to review, personalize with the customer's name, and send.
    """
    return f"""Subject: We'd love to keep you with us — a quick note

Dear Customer,

We noticed a few things about your account that made us want to reach out personally. As a valued member of our ({segment})customer group, your experience with us matters a lot.

We understand that factors like {key_reasons} can sometimes make a service feel like it's not the perfect fit anymore — and we'd genuinely like the chance to make it right.

Based on your account, our team has flagged this as: {priority}

We'd love to offer a personalized plan review or a loyalty discount to make sure you're getting the best value. Could we schedule a quick call this week?

Warm regards,
Customer Retention Team

---
[This is an AI-generated draft. Please review, personalize with the customer's actual name and contact details, and approve before sending.]"""


def run_churn_analyst_agent(segment_name, risk, churn_prob, shap_df, monthly_charges):
    """Orchestrates the Churn Analyst Agent: gives Gemini the prediction context
    and lets it decide, via tool calling, whether retention action is warranted
    and whether to draft an email."""
    if genai_client is None:
        return "GOOGLE_API_KEY not found. Add it to your .env file to enable the agent."

    top_reasons = ", ".join([
        format_feature_label(f, v) for f, v in zip(shap_df['Feature'], shap_df['Value'])
    ][-4:])

    prompt = f"""You are a Churn Analyst Agent helping a retention team.

Customer data:
- Behavioral segment: {segment_name}
- Churn risk level: {risk}
- Churn probability: {churn_prob:.1%}
- Monthly charges: ₹{monthly_charges}
- Top factors influencing this prediction (from SHAP): {top_reasons}

Your task:
1. In 2-3 sentences, explain in plain language why this customer has this risk level, referencing the SHAP factors.
2. Call check_retention_priority to determine the recommended action.
3. If the priority indicates action is needed, call draft_retention_email to produce a draft.
4. Give a short final summary of your recommendation.
"""

    try:
        response = genai_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config={"tools": [check_retention_priority, draft_retention_email]}
        )
    except Exception as e:
        return f"Agent call failed with an error:\n\n```\n{type(e).__name__}: {e}\n```"

    text = response.text

    if not text:
        # Some models occasionally return an empty final text after a tool call.
        # Fall back to manually extracting text from the response, and if that's
        # also empty, surface the raw response so the issue can be diagnosed.
        try:
            parts = response.candidates[0].content.parts
            text = "\n".join(p.text for p in parts if getattr(p, "text", None))
        except Exception:
            text = ""

    if not text:
        return (
            "The agent call succeeded but returned an empty response "
            "(a known issue with some Gemini model versions after a tool call). "
            f"Finish reason: {getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'no candidates'}. "
            "Try clicking Run again, or switch the model to 'gemini-3.6-pro' in the code."
        )

    return text


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">📊 Churn Pilot AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter a customer\'s details to see which behavioral segment they belong to and their churn risk.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Account Details")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges (₹)", 18, 120, 70)
    total_charges = st.number_input("Total Charges (₹)", min_value=0, value=800)
    phone_service = st.radio("Phone Service", ['Yes', 'No'], horizontal=True)
    multiple_lines = st.radio("Multiple Lines", ['No phone service', 'No', 'Yes'], horizontal=True)
    paperless_billing = st.radio("Paperless Billing", ['Yes', 'No'], horizontal=True)
    payment_method = st.selectbox(
        "Payment Method",
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    )
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])

with col2:
    st.subheader("Services Used")
    internet_service = st.radio("Internet Service", ['DSL', 'Fiber optic', 'No'], horizontal=True)
    online_security = st.radio("Online Security", ['No', 'Yes', 'No internet service'], horizontal=True)
    online_backup = st.radio("Online Backup", ['No', 'Yes', 'No internet service'], horizontal=True)
    device_protection = st.radio("Device Protection", ['No', 'Yes', 'No internet service'], horizontal=True)
    tech_support = st.radio("Tech Support", ['No', 'Yes', 'No internet service'], horizontal=True)
    streaming_tv = st.radio("Streaming TV", ['No', 'Yes', 'No internet service'], horizontal=True)
    streaming_movies = st.radio("Streaming Movies", ['No', 'Yes', 'No internet service'], horizontal=True)

st.write("")
predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Prediction output
# ---------------------------------------------------------
if predict_clicked:
    segment_name, risk, churn_prob, classification_input = predict_customer(
        tenure, phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges
    )
    # Store everything needed to redraw the results, so it survives the rerun
    # that happens when the "Run Churn Analyst Agent" button is clicked below.
    st.session_state['prediction'] = {
        'segment_name': segment_name,
        'risk': risk,
        'churn_prob': churn_prob,
        'classification_input': classification_input,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
    }
    # A fresh prediction invalidates any previous agent response.
    st.session_state.pop('agent_response', None)

if 'prediction' in st.session_state:
    p = st.session_state['prediction']
    segment_name = p['segment_name']
    risk = p['risk']
    churn_prob = p['churn_prob']
    classification_input = p['classification_input']
    tenure = p['tenure']
    monthly_charges = p['monthly_charges']

    risk_class_map = {"High Risk": "risk-high", "Medium Risk": "risk-medium", "Low Risk": "risk-low"}
    emoji_map = {"High Risk": "⚠️", "Medium Risk": "🟡", "Low Risk": "✅"}
    risk_class = risk_class_map[risk]
    emoji = emoji_map[risk]

    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div class="risk-title">{emoji} {risk}</div>
        <div style="font-size:0.85rem; opacity:0.8; margin-top:0.2rem;">(this customer's individual churn prediction)</div>
        <div style="margin-top:0.8rem;">Behavioral Segment:</div>
        <div class="segment-badge">{segment_name}</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ---------------- Row 1: SHAP explanation + Segment reasoning ----------------
    shap_df, base_value = get_shap_reasons(classification_input, top_n=6)
    predicted_key = [k for k, v in SEGMENT_NAMES.items() if v == segment_name][0]
    seg_stats = SEGMENT_STATS[predicted_key]

    reason_col1, reason_col2 = st.columns(2)

    with reason_col1:
        with st.container(border=True):
            st.markdown('<div class="card-label">Why this churn risk (SHAP)</div>', unsafe_allow_html=True)
            st.caption("Red = pushes toward churn · Blue = pushes away — specific to this customer.")

            colors = ["#EF4444" if v > 0 else "#3B82F6" for v in shap_df['SHAP Value']]
            labels = [format_feature_label(f, v) for f, v in zip(shap_df['Feature'], shap_df['Value'])]

            shap_fig = go.Figure(go.Bar(
                x=shap_df['SHAP Value'], y=labels, orientation='h', marker_color=colors
            ))
            shap_fig.update_layout(
                height=300, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="SHAP value (impact on churn prediction)"
            )
            st.plotly_chart(shap_fig, use_container_width=True)

    with reason_col2:
        with st.container(border=True):
            st.markdown(f'<div class="card-label">Why the {segment_name} segment</div>', unsafe_allow_html=True)
            st.markdown(f"**{SEGMENT_DESCRIPTIONS[predicted_key]}**")
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("This Customer — Tenure", f"{tenure} mo", delta=f"{tenure - seg_stats['avg_tenure']:.0f} vs avg")
            m2.metric("This Customer — Charges", f"₹{monthly_charges:.0f}", delta=f"{monthly_charges - seg_stats['avg_monthly']:.0f} vs avg")
            st.caption(f"Segment average: {seg_stats['avg_tenure']:.0f} months tenure · ₹{seg_stats['avg_monthly']:.0f} monthly charges")

    st.write("")

    # ---------------- Row 2: Gauge + Segment distribution pie ----------------
    dash_col1, dash_col2 = st.columns(2)

    with dash_col1:
        with st.container(border=True):
            st.markdown('<div class="card-label">Churn Probability</div>', unsafe_allow_html=True)
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                number={'suffix': "%", 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#4338CA"},
                    'steps': [
                        {'range': [0, 35], 'color': "#DCFCE7"},
                        {'range': [35, 60], 'color': "#FEF3C7"},
                        {'range': [60, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.8, 'value': 60}
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(t=10, b=10, l=20, r=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

    with dash_col2:
        with st.container(border=True):
            st.markdown('<div class="card-label">Customer Base by Segment</div>', unsafe_allow_html=True)
            pie_labels = [SEGMENT_NAMES[i] for i in range(4)]
            pie_values = [SEGMENT_STATS[i]["share"] for i in range(4)]
            pull = [0.1 if i == predicted_key else 0 for i in range(4)]

            pie_fig = px.pie(
                names=pie_labels, values=pie_values,
                color_discrete_sequence=["#4338CA", "#6366F1", "#818CF8", "#A5B4FC"],
                hole=0.4
            )
            pie_fig.update_traces(pull=pull, textinfo='label+percent')
            pie_fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(pie_fig, use_container_width=True)

    st.write("")

    # ---------------- Row 3: Tenure + Charges comparison ----------------
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        with st.container(border=True):
            st.markdown('<div class="card-label">Tenure vs Segment Avg</div>', unsafe_allow_html=True)
            tenure_fig = go.Figure(data=[
                go.Bar(name='This Customer', x=['Tenure (months)'], y=[tenure], marker_color="#4338CA"),
                go.Bar(name='Segment Avg', x=['Tenure (months)'], y=[seg_stats["avg_tenure"]], marker_color="#C7D2FE")
            ])
            tenure_fig.update_layout(height=240, barmode='group', margin=dict(t=10, b=10, l=10, r=10),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(tenure_fig, use_container_width=True)

    with comp_col2:
        with st.container(border=True):
            st.markdown('<div class="card-label">Charges vs Segment Avg</div>', unsafe_allow_html=True)
            charge_fig = go.Figure(data=[
                go.Bar(name='This Customer', x=['Monthly Charges (₹)'], y=[monthly_charges], marker_color="#4338CA"),
                go.Bar(name='Segment Avg', x=['Monthly Charges (₹)'], y=[seg_stats["avg_monthly"]], marker_color="#C7D2FE")
            ])
            charge_fig.update_layout(height=240, barmode='group', margin=dict(t=10, b=10, l=10, r=10),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(charge_fig, use_container_width=True)

    st.write("")

    # ---------------- Row 4: Top churn drivers (global feature importance) ----------------
    with st.container(border=True):
        st.markdown('<div class="card-label">Top Churn Drivers — Model Feature Importance</div>', unsafe_allow_html=True)
        importances = churn_model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': [format_feature_name(c) for c in CLASSIFICATION_COLS],
            'Importance': importances
        }).sort_values('Importance', ascending=True).tail(8)

        fi_fig = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Purples'
        )
        fi_fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fi_fig, use_container_width=True)

    st.write("")

    # ---------------- Row 5: Churn Analyst Agent ----------------
    with st.container(border=True):
        st.markdown('<div class="card-label">🤖 Churn Analyst Agent</div>', unsafe_allow_html=True)
        st.caption("An AI agent that reasons over this prediction, decides if retention action is needed, and can draft an outreach email — using tool calling, not just text generation.")

        if st.button("Run Churn Analyst Agent", use_container_width=True):
            with st.spinner("Agent is reasoning over this customer's data..."):
                agent_response = run_churn_analyst_agent(
                    segment_name, risk, churn_prob, shap_df, monthly_charges
                )
            st.session_state['agent_response'] = agent_response

        if 'agent_response' in st.session_state:
            st.markdown(st.session_state['agent_response'])

st.markdown("---")
st.caption("Built with Streamlit · Segmentation via PCA + K-Means · Churn Prediction via tuned XGBoost")