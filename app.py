# m3_ins_analytics_app.py
# Streamlit app for exploring insurance charges with interactive charts and maps
# Author: (Your Name)
# Date: 2025-10-16

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ---------------------------
# Page config & minimalist styling
# ---------------------------
st.set_page_config(
    page_title="M3 Insurance – Health Charges Explorer",
    page_icon="📊",
    layout="wide",
)

# Minimal, clean CSS tweaks
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
        .metric-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
        .small-note { color: #6b7280; font-size: 0.9rem; }
        .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f4f6; margin-right:6px; }
        .sidebar .sidebar-content { background: #fafafa; }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] { background: #f7f7f8; border-radius: 12px; padding: 10px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers & data
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file: io.BytesIO | None = None) -> pd.DataFrame:
    """Load the insurance dataset.
    Priority: uploaded file -> local 'insurance.csv' -> fallback to demo generated.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Try local
        candidates = [
            "insurance.csv",
            "data/insurance.csv",
            "./insurance 3.csv",  # allow files with spaces
            "./insurance_3.csv",
        ]
        df = None
        for path in candidates:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        if df is None:
            # As a last resort, synthesize a small demo dataset with similar schema
            np.random.seed(7)
            n = 500
            df = pd.DataFrame({
                "age": np.random.randint(18, 65, n),
                "sex": np.random.choice(["male", "female"], n),
                "bmi": np.round(np.random.normal(30, 6, n), 1),
                "children": np.random.poisson(1.1, n).clip(0, 5),
                "smoker": np.random.choice(["yes", "no"], n, p=[0.2, 0.8]),
                "region": np.random.choice(["northeast", "southeast", "southwest", "northwest"], n),
            })
            base = 2500 + df["age"] * 60 + (df["bmi"] - 25) * 120 + df["children"] * 300
            df["charges"] = np.where(df["smoker"] == "yes", base * 2.5, base) + np.random.normal(0, 1200, n)
    # Clean types
    df = df.dropna(subset=["age", "sex", "bmi", "children", "smoker", "region", "charges"]).copy()
    df["sex"] = df["sex"].str.lower().str.strip()
    df["smoker"] = df["smoker"].str.lower().str.strip()
    df["region"] = df["region"].str.lower().str.replace("_", " ").str.strip()
    # Clip BMI to realistic range
    df["bmi"] = df["bmi"].clip(10, 60)
    return df

REGION_CENTROIDS = {
    "northeast": {"lat": 42.5, "lon": -75.0},  # approximate region centroids
    "southeast": {"lat": 33.0, "lon": -84.0},
    "southwest": {"lat": 34.0, "lon": -112.0},
    "northwest": {"lat": 45.0, "lon": -120.0},
}

@st.cache_data(show_spinner=False)
def add_synthetic_geo(df: pd.DataFrame, jitter_km: float = 250.0) -> pd.DataFrame:
    """Map regions to approximate lat/lon with light jitter for visualization.
    NOTE: The dataset has no geocoordinates; this map is illustrative, not precise.
    """
    df = df.copy()
    rng = np.random.default_rng(42)

    def jitter_latlon(row):
        r = REGION_CENTROIDS.get(row["region"], {"lat": 39.5, "lon": -98.35})
        # ~1 degree lat ≈ 111 km, lon scaling by cos(lat)
        lat_j = (jitter_km / 111.0) * rng.normal(0, 0.5)
        lon_scale = max(np.cos(np.radians(r["lat"])), 0.3)
        lon_j = (jitter_km / (111.0 * lon_scale)) * rng.normal(0, 0.5)
        return pd.Series({"lat": r["lat"] + lat_j, "lon": r["lon"] + lon_j})

    df[["lat", "lon"]] = df.apply(jitter_latlon, axis=1)
    return df

# ---------------------------
# Sidebar – filters and data upload
# ---------------------------
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload CSV (columns: age, sex, bmi, children, smoker, region, charges)", type=["csv"]) 

df = load_data(uploaded)

with st.sidebar.expander("Filter data", expanded=True):
    age_range = st.slider("Age", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
    bmi_range = st.slider("BMI", float(df["bmi"].min()), float(df["bmi"].max()), (float(df["bmi"].min()), float(df["bmi"].max())))
    children_sel = st.multiselect("Children", options=sorted(df["children"].unique()), default=sorted(df["children"].unique()))
    sex_sel = st.multiselect("Sex", options=sorted(df["sex"].unique()), default=sorted(df["sex"].unique()))
    smoker_sel = st.multiselect("Smoker", options=sorted(df["smoker"].unique()), default=sorted(df["smoker"].unique()))
    region_sel = st.multiselect("Region", options=sorted(df["region"].unique()), default=sorted(df["region"].unique()))

filtered = df[
    (df["age"].between(age_range[0], age_range[1])) &
    (df["bmi"].between(bmi_range[0], bmi_range[1])) &
    (df["children"].isin(children_sel)) &
    (df["sex"].isin(sex_sel)) &
    (df["smoker"].isin(smoker_sel)) &
    (df["region"].isin(region_sel))
].copy()

st.sidebar.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** rows")

# Download filtered
@st.cache_data(show_spinner=False)
def to_csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    label="⬇️ Download filtered CSV",
    data=to_csv_bytes(filtered),
    file_name="filtered_insurance.csv",
    mime="text/csv",
)

# ---------------------------
# Header
# ---------------------------
st.title("📊 M3 Insurance – Health Charges Analytics")
st.caption("Interactive exploration of medical charges by demographics, lifestyle, and region. Minimal, fast, and interview-ready.")

# KPI cards
kpi1 = float(filtered["charges"].mean()) if len(filtered) else 0
kpi2 = float(filtered["bmi"].mean()) if len(filtered) else 0
kpi3 = float((filtered["smoker"] == "yes").mean()*100) if len(filtered) else 0
kpi4 = float(filtered["children"].mean()) if len(filtered) else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Charges ($)", f"{kpi1:,.0f}")
col2.metric("Avg BMI", f"{kpi2:,.1f}")
col3.metric("% Smokers", f"{kpi3:.1f}%")
col4.metric("Avg # Children", f"{kpi4:.2f}")

st.write("")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore", "Map", "Model"])

# Overview Tab
with tab1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Distributions & Segments")
        sel_color = st.selectbox("Color by", options=["smoker", "sex", "region", None], index=0)
        color_kw = {"color": sel_color} if sel_color else {}
        fig1 = px.histogram(filtered, x="charges", nbins=40, **color_kw, marginal="box")
        fig1.update_layout(margin=dict(l=0,r=0,b=0,t=30), height=380)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.box(filtered, x="region", y="charges", color="smoker" if sel_color=="smoker" else None)
        fig2.update_layout(margin=dict(l=0,r=0,b=0,t=30), height=380)
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.subheader("Correlations")
        num_df = filtered[["age", "bmi", "children", "charges"]].corr().round(2)
        fig_corr = px.imshow(num_df, text_auto=True, aspect="auto")
        fig_corr.update_layout(margin=dict(l=0,r=0,b=0,t=30), height=350)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown(
            """
            **Reading tips:**
            - Smokers typically exhibit much higher charges.
            - Charges generally rise with age, BMI, and number of dependents.
            - Use the filters to test hypotheses for specific cohorts.
            """
        )

# Explore Tab
with tab2:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Scatter Explorer")
        x = st.selectbox("X-axis", ["age", "bmi", "children"], index=1)
        y = st.selectbox("Y-axis", ["charges", "bmi", "age"], index=0)
        size_by = st.selectbox("Bubble size", [None, "charges", "bmi", "age", "children"], index=1)
        color_by = st.selectbox("Color", [None, "smoker", "sex", "region"], index=1)
        fig_sc = px.scatter(filtered, x=x, y=y, size=size_by, color=color_by, hover_data=["sex", "smoker", "region", "children"], trendline=None)
        fig_sc.update_layout(margin=dict(l=0,r=0,b=0,t=30), height=460)
        st.plotly_chart(fig_sc, use_container_width=True)
    with right:
        st.subheader("Facet by Region")
        facet_var = st.selectbox("Facet column", [None, "region", "smoker", "sex"], index=1)
        if facet_var:
            fig_fc = px.histogram(filtered, x="bmi", color="smoker", facet_col=facet_var, facet_col_wrap=2, nbins=30)
            fig_fc.update_layout(margin=dict(l=0,r=0,b=0,t=30), height=460)
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("Pick a facet to compare distributions across groups.")

# Map Tab
with tab3:
    st.subheader("Regional Map (Illustrative)")
    st.caption("This dataset contains region labels only. Points are plotted at approximate regional centroids with light jitter for pattern-finding – not for exact locations.")
    geo = add_synthetic_geo(filtered)
    color_map = {"yes": [200, 55, 45], "no": [34, 139, 230]}
    geo["color"] = geo["smoker"].map(color_map).fillna([120,120,120])
    geo["size"] = np.interp(geo["charges"], (geo["charges"].min(), geo["charges"].max()), (200, 2000))

    view_state = pdk.ViewState(latitude=39.5, longitude=-98.35, zoom=3.2, pitch=0)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=geo,
        get_position='[lon, lat]',
        get_radius="size",
        get_fill_color="color",
        pickable=True,
        opacity=0.35,
    )

    tooltip = {
        "html": "<b>Region:</b> {region}<br/><b>Smoker:</b> {smoker}<br/><b>Age:</b> {age}<br/><b>BMI:</b> {bmi}<br/><b>Charges:</b> ${charges}",
        "style": {"backgroundColor": "#111", "color": "white"}
    }

    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(r, use_container_width=True)

    # Region-level summary
    st.markdown("### Region Summary")
    reg_summary = geo.groupby("region").agg(
        records=("charges", "size"),
        avg_charges=("charges", "mean"),
        smoker_rate=("smoker", lambda s: (s=="yes").mean()),
        avg_bmi=("bmi", "mean"),
    ).reset_index()
    reg_summary["smoker_rate"] = (reg_summary["smoker_rate"]*100).round(1)
    reg_summary["avg_charges"] = reg_summary["avg_charges"].round(0)
    st.dataframe(reg_summary, use_container_width=True)

# Model Tab
with tab4:
    st.subheader("Predictive Model – What-if Charges")
    st.caption("Train a quick model on the current filtered cohort, then run what-if scenarios with the form below.")

    algo = st.radio("Algorithm", ["Random Forest", "Linear Regression"], horizontal=True)

    # Prepare data
    if len(filtered) < 50:
        st.warning("Not enough rows in the filtered set for a stable model. Widen your filters for better results.")

    X = filtered[["age", "bmi", "children", "sex", "smoker", "region"]]
    y = filtered["charges"]

    cat_feats = ["sex", "smoker", "region"]
    num_feats = ["age", "bmi", "children"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ("num", "passthrough", num_feats),
    ])

    if algo == "Random Forest":
        model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1, max_depth=None, min_samples_leaf=2)
    else:
        model = LinearRegression()

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    # Train/validate
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        m1, m2 = st.columns(2)
        m1.metric("R² (test)", f"{r2:.3f}")
        m2.metric("MAE (test)", f"${mae:,.0f}")

        # Feature importances (for RF) or coefficients (for LR)
        st.markdown("### Feature Importance / Coefficients")
        if algo == "Random Forest":
            # Extract feature names after one-hot
            ohe = pipe.named_steps["pre"].named_transformers_["cat"]
            cat_names = list(ohe.get_feature_names_out(cat_feats))
            feat_names = cat_names + num_feats
            importances = pipe.named_steps["model"].feature_importances_
            fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False).head(20)
            fig_fi = px.bar(fi, x="importance", y="feature", orientation="h")
        else:
            ohe = pipe.named_steps["pre"].named_transformers_["cat"]
            cat_names = list(ohe.get_feature_names_out(cat_feats))
            feat_names = cat_names + num_feats
            coefs = pipe.named_steps["model"].coef_
            fi = pd.DataFrame({"feature": feat_names, "importance": coefs}).sort_values("importance", ascending=False)
            fig_fi = px.bar(fi, x="importance", y="feature", orientation="h")
        fig_fi.update_layout(margin=dict(l=0,r=0,b=0,t=10), height=420)
        st.plotly_chart(fig_fi, use_container_width=True)

    except Exception as e:
        st.error(f"Model training failed: {e}")

    st.markdown("### What-if Prediction")
    with st.form("what_if"):
        c1, c2, c3 = st.columns(3)
        age_in = c1.slider("Age", 18, 64, 35)
        bmi_in = c2.slider("BMI", 12.0, 60.0, 28.0, step=0.1)
        children_in = c3.selectbox("Children", options=sorted(df["children"].unique()), index=0)
        c4, c5, c6 = st.columns(3)
        sex_in = c4.selectbox("Sex", ["female", "male"], index=0)
        smoker_in = c5.selectbox("Smoker", ["no", "yes"], index=0)
        region_in = c6.selectbox("Region", sorted(df["region"].unique()))
        submitted = st.form_submit_button("Predict charges →")

    if submitted:
        try:
            sample = pd.DataFrame([{ 
                "age": age_in,
                "bmi": bmi_in,
                "children": int(children_in),
                "sex": sex_in,
                "smoker": smoker_in,
                "region": region_in,
            }])
            pred = float(pipe.predict(sample)[0])
            st.success(f"Estimated charges: **${pred:,.0f}** for the selected profile.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------
# Footer
# ---------------------------
st.write("")
st.markdown(
    """
    <div class='small-note'>
    Built with Streamlit, Plotly, and PyDeck. Map points use approximate regional centroids with random jitter for visualization only.
    </div>
    """,
    unsafe_allow_html=True,
)
