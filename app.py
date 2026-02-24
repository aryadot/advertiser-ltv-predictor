import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import io

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer LTV Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0a0e0a; color: #e0e0e0; }

    section[data-testid="stSidebar"] {
        background-color: #0f150f;
        border-right: 1px solid #1a2e1a;
    }

    .kpi-card {
        background: #0f150f;
        border: 1px solid #1a2e1a;
        border-left: 3px solid #10b981;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .kpi-card h4 {
        color: #6b7280; font-size: 11px; font-weight: 600;
        letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 6px;
    }
    .kpi-card h2 { color: #ffffff; font-size: 26px; font-weight: 700; margin: 0; }
    .kpi-card p { color: #10b981; font-size: 12px; margin: 4px 0 0 0; }

    .section-title {
        font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
        text-transform: uppercase; color: #10b981;
        margin-bottom: 14px; padding-bottom: 6px;
        border-bottom: 1px solid #1a2e1a;
    }

    .tier-high {
        background: #052015; border: 1px solid #10b981;
        border-radius: 10px; padding: 24px; text-align: center;
    }
    .tier-mid {
        background: #1a1500; border: 1px solid #d97706;
        border-radius: 10px; padding: 24px; text-align: center;
    }
    .tier-low {
        background: #150505; border: 1px solid #ef4444;
        border-radius: 10px; padding: 24px; text-align: center;
    }

    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
    }
    .badge-green { background: #052015; color: #10b981; border: 1px solid #10b981; }
    .badge-yellow { background: #1a1500; color: #d97706; border: 1px solid #d97706; }
    .badge-red { background: #150505; color: #ef4444; border: 1px solid #ef4444; }

    .alert-danger {
        background: #150505; border: 1px solid #ef4444;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
        color: #ef4444; font-size: 13px;
    }
    .alert-warning {
        background: #1a1500; border: 1px solid #d97706;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
        color: #d97706; font-size: 13px;
    }
    .alert-success {
        background: #052015; border: 1px solid #10b981;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
        color: #10b981; font-size: 13px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #059669, #10b981);
        color: #000000; border: none; border-radius: 8px;
        padding: 10px 20px; font-size: 13px; font-weight: 700;
        width: 100%; letter-spacing: 0.5px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stTabs [data-baseweb="tab-list"] {
        background: #0f150f; border-radius: 8px;
        padding: 3px; border: 1px solid #1a2e1a;
    }
    .stTabs [data-baseweb="tab"] { color: #6b7280; font-weight: 500; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #10b981 !important; color: #000000 !important; font-weight: 700 !important; }

    label { color: #9ca3af !important; font-size: 12px !important; font-weight: 500 !important; }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        background-color: #0f150f !important;
        border: 1px solid #1a2e1a !important;
        border-radius: 6px !important;
        color: #e0e0e0 !important;
        font-size: 13px !important;
    }

    div[data-testid="stSidebarContent"] label {
        color: #9ca3af !important;
    }

    .pulse-container {
        position: relative; width: 80px; height: 80px;
        margin: 0 auto 20px auto;
    }
    .pulse-ring {
        position: absolute; border-radius: 50%;
        border: 1.5px solid #10b981;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        animation: pulse 2.4s ease-out infinite; opacity: 0;
    }
    .pulse-ring:nth-child(1) { animation-delay: 0s; }
    .pulse-ring:nth-child(2) { animation-delay: 0.8s; }
    .pulse-ring:nth-child(3) { animation-delay: 1.6s; }
    @keyframes pulse {
        0%   { width: 10px; height: 10px; opacity: 0.9; }
        100% { width: 75px; height: 75px; opacity: 0; }
    }
    .pulse-core {
        position: absolute; width: 10px; height: 10px;
        border-radius: 50%; background: #10b981;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 10px #10b981;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('model/latest_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('model/shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('model/drift_baseline.json', 'r') as f:
        baseline = json.load(f)
    with open('model/feature_cols.json', 'r') as f:
        feature_cols = json.load(f)
    with open('model/model_registry.json', 'r') as f:
        registry = json.load(f)
    with open('model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return pipeline, explainer, baseline, feature_cols, registry, le

pipeline, explainer, baseline, feature_cols, registry, le = load_artifacts()
latest = registry[-1]

# ── Helpers ──────────────────────────────────────────────────
def validate_input(input_dict):
    alerts = []
    for col, val in input_dict.items():
        if col not in baseline:
            continue
        b = baseline[col]
        z = abs((val - b['mean']) / (b['std'] + 1e-9))
        if z > 3:
            alerts.append(('danger', f"{col}: {val:.2f} is {z:.1f}σ from training mean ({b['mean']:.2f})"))
        elif z > 2:
            alerts.append(('warning', f"{col}: {val:.2f} is outside normal range (mean: {b['mean']:.2f})"))
    return alerts

def predict_ltv(input_dict):
    input_df = pd.DataFrame([input_dict])[feature_cols]
    log_pred = pipeline.predict(input_df)[0]
    return np.expm1(log_pred), log_pred, input_df

def get_tier(ltv):
    if ltv >= 500:
        return "High Value", "#10b981", "tier-high", "badge-green"
    elif ltv >= 150:
        return "Mid Value", "#d97706", "tier-mid", "badge-yellow"
    else:
        return "Low Value", "#ef4444", "tier-low", "badge-red"

def detect_drift(input_df):
    rows = []
    for col in feature_cols:
        if col not in baseline:
            continue
        b = baseline[col]
        col_mean = input_df[col].mean()
        z = abs((col_mean - b['mean']) / (b['std'] + 1e-9))
        rows.append({
            'Feature': col,
            'Input Mean': round(col_mean, 3),
            'Training Mean': round(b['mean'], 3),
            'Deviation (σ)': round(z, 2),
            'Status': 'DRIFT' if z > 2 else 'OK'
        })
    return pd.DataFrame(rows)

def mpl_dark(fig, axes_list):
    fig.patch.set_facecolor('#0a0e0a')
    for ax in axes_list:
        ax.set_facecolor('#0f150f')
        ax.tick_params(colors='#6b7280')
        ax.xaxis.label.set_color('#6b7280')
        ax.yaxis.label.set_color('#6b7280')
        ax.title.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a2e1a')

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#10b981;font-size:18px;font-weight:700;margin-bottom:4px;'>LTV Intelligence</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4b5563;font-size:11px;margin-bottom:24px;'>Customer Value Platform</p>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model Status</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='kpi-card'>
        <h4>Active Version</h4>
        <h2 style='font-size:14px;'>{latest['version']}</h2>
        <p>R² {latest['r2']} — MAE ${latest['mae_dollars']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='kpi-card'>
        <h4>Training Size</h4>
        <h2>{latest['n_train']:,}</h2>
        <p>customers</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='kpi-card'>
        <h4>Model Type</h4>
        <h2 style='font-size:14px;'>{latest['model_type']}</h2>
        <p>sklearn Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>LTV Tiers</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#6b7280; line-height:2;'>
        <span class='badge badge-green'>High Value</span> &nbsp; $500+<br>
        <span class='badge badge-yellow'>Mid Value</span> &nbsp;&nbsp; $150–$500<br>
        <span class='badge badge-red'>Low Value</span> &nbsp;&nbsp; Under $150
    </div>
    """, unsafe_allow_html=True)

# ── Main header ───────────────────────────────────────────────
st.markdown("<h1 style='color:#ffffff;font-size:28px;font-weight:700;margin-bottom:2px;'>Customer LTV Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#1a2e1a;margin:12px 0 24px 0;'>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Score Customer",
    "Batch Scoring",
    "Model Registry",
    "Drift Monitor"
])

# ════════════════════════════════════════════════
# TAB 1 — Score Customer
# ════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("<div class='section-title'>RFM Signals</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            recency_days           = st.number_input("Recency (Days)", min_value=0, value=30)
            frequency              = st.number_input("Order Frequency", min_value=1, value=2)
            total_items            = st.number_input("Total Items", min_value=1, value=3)
            orders_per_month       = st.number_input("Orders / Month", min_value=0.0, value=0.5, step=0.1)
        with c2:
            avg_item_price         = st.number_input("Avg Item Price ($)", min_value=0.0, value=80.0)
            unique_products        = st.number_input("Unique Products", min_value=1, value=2)
            unique_sellers         = st.number_input("Unique Sellers", min_value=1, value=2)
            customer_lifetime_days = st.number_input("Lifetime (Days)", min_value=0, value=90)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Behavioral Signals</div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            avg_installments  = st.number_input("Avg Installments", min_value=1.0, value=2.0, step=0.5)
            avg_review_score  = st.number_input("Avg Review Score", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        with b2:
            freight_ratio = st.slider("Freight Ratio", min_value=0.0, max_value=2.0, value=0.2, step=0.01,
                                       help="Freight cost / item price")
            state = st.selectbox("State", sorted(le.classes_))

        st.markdown("<br>", unsafe_allow_html=True)
        score_btn = st.button("Score Customer")

    with right:
        if not score_btn:
            st.markdown("""
            <div style='height:500px; background:#0f150f; border-radius:12px;
                        border:1px solid #1a2e1a; padding-top:160px; text-align:center;'>
                <div class='pulse-container'>
                    <div class='pulse-ring'></div>
                    <div class='pulse-ring'></div>
                    <div class='pulse-ring'></div>
                    <div class='pulse-core'></div>
                </div>
                <div style='color:#ffffff;font-size:15px;font-weight:600;margin-bottom:8px;'>
                    Awaiting Input
                </div>
                <div style='color:#4b5563;font-size:12px;max-width:200px;margin:0 auto;line-height:1.7;'>
                    Fill in customer signals and click Score Customer
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            try:
                state_encoded = le.transform([state])[0]
            except:
                state_encoded = 0

            input_dict = {
                'recency_days':           recency_days,
                'frequency':              frequency,
                'total_items':            total_items,
                'avg_item_price':         avg_item_price,
                'freight_ratio':          freight_ratio,
                'unique_products':        unique_products,
                'unique_sellers':         unique_sellers,
                'avg_installments':       avg_installments,
                'avg_review_score':       avg_review_score,
                'customer_lifetime_days': customer_lifetime_days,
                'orders_per_month':       orders_per_month,
                'state_encoded':          state_encoded
            }

            alerts = validate_input(input_dict)
            if alerts:
                st.markdown("<div class='section-title'>Validation Alerts</div>", unsafe_allow_html=True)
                for level, msg in alerts:
                    st.markdown(f"<div class='alert-{level}'>{msg}</div>", unsafe_allow_html=True)

            ltv, ltv_log, input_df = predict_ltv(input_dict)
            tier_label, tier_color, tier_class, badge_class = get_tier(ltv)

            st.markdown("<div class='section-title'>LTV Score</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='{tier_class}' style='margin-bottom:20px;'>
                <div style='font-size:11px;font-weight:700;letter-spacing:1.5px;
                            text-transform:uppercase;color:{tier_color};margin-bottom:6px;'>
                    {tier_label} Customer
                </div>
                <div style='font-size:72px;font-weight:800;color:{tier_color};
                            margin:8px 0;line-height:1;'>
                    ${ltv:.0f}
                </div>
                <div style='font-size:12px;color:#6b7280;margin-top:8px;'>
                    Predicted Lifetime Value
                </div>
            </div>
            """, unsafe_allow_html=True)

            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"""<div class='kpi-card'>
                    <h4>Tier</h4>
                    <h2 style='font-size:16px;color:{tier_color};'>{tier_label}</h2>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class='kpi-card'>
                    <h4>Log LTV</h4>
                    <h2>{ltv_log:.3f}</h2>
                </div>""", unsafe_allow_html=True)
            with k3:
                percentile = int(min(99, max(1, (ltv / 1679) * 100)))
                st.markdown(f"""<div class='kpi-card'>
                    <h4>Est. Percentile</h4>
                    <h2>{percentile}th</h2>
                </div>""", unsafe_allow_html=True)

            # SHAP
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Feature Attribution (SHAP)</div>", unsafe_allow_html=True)

            scaler = pipeline.named_steps['scaler']
            input_scaled = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(input_scaled, columns=feature_cols)
            shap_vals = explainer.shap_values(input_scaled_df)

            fig, ax = plt.subplots(figsize=(8, 5))
            mpl_dark(fig, [ax])
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=explainer.expected_value,
                    data=input_scaled_df.iloc[0],
                    feature_names=feature_cols
                ), show=False
            )
            plt.tight_layout()
            st.pyplot(fig)

# ════════════════════════════════════════════════
# TAB 2 — Batch Scoring
# ════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Batch Customer Scoring</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;font-size:13px;margin-bottom:20px;'>Upload a CSV of customer metrics to score multiple customers at once. Download the template to see required format.</p>", unsafe_allow_html=True)

    col_dl, col_empty = st.columns([1, 3])
    with col_dl:
        template_df = pd.DataFrame(columns=feature_cols)
        template_df.loc[0] = [30, 2, 3, 80.0, 0.2, 2, 2, 2.0, 4.0, 90, 0.5, 5]
        buf = io.StringIO()
        template_df.to_csv(buf, index=False)
        st.download_button("Download Template CSV", buf.getvalue(),
                           "ltv_template.csv", "text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Customer CSV", type=['csv'])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in batch_df.columns]

        if missing:
            st.markdown(f"<div class='alert-danger'>Missing columns: {missing}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-success'>Loaded {len(batch_df):,} customers successfully.</div>", unsafe_allow_html=True)

            X_batch = batch_df[feature_cols]
            batch_df['predicted_ltv'] = np.expm1(pipeline.predict(X_batch))
            batch_df['ltv_tier'] = batch_df['predicted_ltv'].apply(
                lambda x: 'High Value' if x >= 500 else ('Mid Value' if x >= 150 else 'Low Value')
            )

            drift_df = detect_drift(X_batch)
            drift_count = (drift_df['Status'] == 'DRIFT').sum()
            if drift_count > 0:
                st.markdown(f"<div class='alert-warning'>{drift_count} features show distribution drift in this batch — predictions may be less reliable.</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Scoring Summary</div>", unsafe_allow_html=True)

            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                st.markdown(f"""<div class='kpi-card'><h4>Total</h4><h2>{len(batch_df):,}</h2><p>customers scored</p></div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""<div class='kpi-card'><h4>Avg LTV</h4><h2>${batch_df['predicted_ltv'].mean():.0f}</h2><p>predicted</p></div>""", unsafe_allow_html=True)
            with s3:
                high = (batch_df['ltv_tier'] == 'High Value').sum()
                st.markdown(f"""<div class='kpi-card'><h4>High Value</h4><h2 style='color:#10b981;'>{high}</h2><p>{high/len(batch_df)*100:.1f}% of batch</p></div>""", unsafe_allow_html=True)
            with s4:
                mid = (batch_df['ltv_tier'] == 'Mid Value').sum()
                st.markdown(f"""<div class='kpi-card'><h4>Mid Value</h4><h2 style='color:#d97706;'>{mid}</h2><p>{mid/len(batch_df)*100:.1f}% of batch</p></div>""", unsafe_allow_html=True)
            with s5:
                low = (batch_df['ltv_tier'] == 'Low Value').sum()
                st.markdown(f"""<div class='kpi-card'><h4>Low Value</h4><h2 style='color:#ef4444;'>{low}</h2><p>{low/len(batch_df)*100:.1f}% of batch</p></div>""", unsafe_allow_html=True)

            # LTV distribution chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Predicted LTV Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 3))
            mpl_dark(fig, [ax])
            ax.hist(batch_df['predicted_ltv'], bins=40, color='#10b981', alpha=0.8, edgecolor='none')
            ax.axvline(150, color='#d97706', linestyle='--', linewidth=1, label='Mid threshold ($150)')
            ax.axvline(500, color='#10b981', linestyle='--', linewidth=1, label='High threshold ($500)')
            ax.legend(facecolor='#0f150f', labelcolor='#9ca3af', fontsize=10)
            ax.set_xlabel('Predicted LTV ($)')
            ax.set_ylabel('Count')
            plt.tight_layout()
            st.pyplot(fig)

            # Results table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Results Table</div>", unsafe_allow_html=True)
            st.dataframe(
                batch_df[['predicted_ltv', 'ltv_tier'] + feature_cols].round(2),
                use_container_width=True
            )

            result_buf = io.StringIO()
            batch_df.to_csv(result_buf, index=False)
            st.download_button("Download Predictions", result_buf.getvalue(),
                               "ltv_predictions.csv", "text/csv")

# ════════════════════════════════════════════════
# TAB 3 — Model Registry
# ════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Production Model</div>", unsafe_allow_html=True)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown(f"""<div class='kpi-card'><h4>Version</h4>
            <h2 style='font-size:13px;'>{latest['version']}</h2><p>current</p></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class='kpi-card'><h4>R²</h4>
            <h2>{latest['r2']}</h2><p>test set</p></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown(f"""<div class='kpi-card'><h4>MAE</h4>
            <h2>${latest['mae_dollars']}</h2><p>in dollars</p></div>""", unsafe_allow_html=True)
    with p4:
        st.markdown(f"""<div class='kpi-card'><h4>RMSE</h4>
            <h2>${latest['rmse_dollars']}</h2><p>in dollars</p></div>""", unsafe_allow_html=True)
    with p5:
        st.markdown(f"""<div class='kpi-card'><h4>Train Size</h4>
            <h2>{latest['n_train']:,}</h2><p>customers</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Feature Set</div>", unsafe_allow_html=True)
    feat_cols = st.columns(4)
    for i, feat in enumerate(feature_cols):
        with feat_cols[i % 4]:
            st.markdown(f"<div style='background:#0f150f;border:1px solid #1a2e1a;border-radius:6px;padding:8px 12px;margin-bottom:8px;font-size:12px;color:#10b981;'>{feat}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Version History</div>", unsafe_allow_html=True)
    registry_df = pd.DataFrame(registry)[[
        'version', 'trained_at', 'model_type', 'r2',
        'mae_dollars', 'rmse_dollars', 'n_train', 'n_test'
    ]]
    registry_df.columns = ['Version', 'Trained At', 'Model', 'R²', 'MAE ($)', 'RMSE ($)', 'Train', 'Test']
    st.dataframe(registry_df, use_container_width=True)

    if len(registry) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Performance Over Versions</div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        mpl_dark(fig, axes)
        versions = [r['version'] for r in registry]
        axes[0].plot(versions, [r['r2'] for r in registry], color='#10b981', marker='o', linewidth=2)
        axes[0].set_title('R² Score')
        axes[0].set_ylim(0, 1)
        axes[1].plot(versions, [r['mae_dollars'] for r in registry], color='#d97706', marker='o', linewidth=2)
        axes[1].set_title('MAE ($)')
        plt.tight_layout()
        st.pyplot(fig)

# ════════════════════════════════════════════════
# TAB 4 — Drift Monitor
# ════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Data Drift Monitor</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;font-size:13px;margin-bottom:20px;'>Upload recent customer data to compare feature distributions against the training baseline. Features beyond 2σ are flagged as drifted.</p>", unsafe_allow_html=True)

    drift_upload = st.file_uploader("Upload Recent Data CSV", type=['csv'], key="drift")

    if drift_upload:
        drift_input = pd.read_csv(drift_upload)
        missing = [c for c in feature_cols if c not in drift_input.columns]

        if missing:
            st.markdown(f"<div class='alert-danger'>Missing columns: {missing}</div>", unsafe_allow_html=True)
        else:
            drift_df = detect_drift(drift_input[feature_cols])
            drift_count = (drift_df['Status'] == 'DRIFT').sum()
            ok_count = (drift_df['Status'] == 'OK').sum()

            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"""<div class='kpi-card'><h4>Features Checked</h4>
                    <h2>{len(drift_df)}</h2><p>total</p></div>""", unsafe_allow_html=True)
            with d2:
                st.markdown(f"""<div class='kpi-card'><h4>Drifted</h4>
                    <h2 style='color:#ef4444;'>{drift_count}</h2><p>features flagged</p></div>""", unsafe_allow_html=True)
            with d3:
                st.markdown(f"""<div class='kpi-card'><h4>Stable</h4>
                    <h2 style='color:#10b981;'>{ok_count}</h2><p>features OK</p></div>""", unsafe_allow_html=True)

            if drift_count == 0:
                st.markdown("<div class='alert-success'>No drift detected — all features within normal distribution range.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-danger'>{drift_count} feature(s) show significant drift. Consider retraining the model.</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Distribution Comparison</div>", unsafe_allow_html=True)

            def style_status(val):
                if val == 'DRIFT':
                    return 'color: #ef4444; font-weight: bold'
                return 'color: #10b981'

            st.dataframe(
                drift_df.style.applymap(style_status, subset=['Status']),
                use_container_width=True
            )

            drifted = drift_df[drift_df['Status'] == 'DRIFT']['Feature'].tolist()
            if drifted:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Drifted Feature Distributions</div>", unsafe_allow_html=True)
                for feat in drifted[:4]:
                    b = baseline[feat]
                    fig, ax = plt.subplots(figsize=(10, 2.5))
                    mpl_dark(fig, [ax])
                    x = np.linspace(b['min'], b['max'], 100)
                    from scipy.stats import norm
                    ax.plot(x, norm.pdf(x, b['mean'], b['std']),
                            color='#10b981', label='Training baseline', linewidth=2)
                    ax.hist(drift_input[feat].dropna(), bins=30, density=True,
                            alpha=0.5, color='#ef4444', label='Incoming data', edgecolor='none')
                    ax.set_title(feat)
                    ax.legend(facecolor='#0f150f', labelcolor='#9ca3af', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
    else:
        st.markdown("""
        <div style='height:280px; background:#0f150f; border-radius:12px;
                    border:1px solid #1a2e1a; padding-top:90px; text-align:center;'>
            <div style='color:#ffffff;font-size:15px;font-weight:600;margin-bottom:8px;'>
                No Data Uploaded
            </div>
            <div style='color:#4b5563;font-size:12px;max-width:260px;margin:0 auto;line-height:1.7;'>
                Upload a CSV of recent customer data to run drift detection
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='color:#1a2e1a;font-size:11px;text-align:center;'>Customer LTV Intelligence Platform — XGBoost + SHAP + MLOps — Marketing ML Portfolio</p>", unsafe_allow_html=True)