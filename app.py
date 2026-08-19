import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path


st.set_page_config(
    page_title="California Home Price Predictor",
    layout="wide"
)


st.markdown(
    """
    <style>
    :root {
        --primary: #1f3a5f;
        --primary-soft: #eaf0f8;
        --accent: #3b6ea8;
        --text-main: #1f2937;
        --text-muted: #6b7280;
        --border: #e5e7eb;
        --background: #f8fafc;
        --card: #ffffff;
    }

    .stApp {
        background: var(--background);
    }

    section[data-testid="stSidebar"] {
        background: #eef3f8;
        border-right: 1px solid var(--border);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    h1, h2, h3 {
        color: var(--text-main);
        letter-spacing: -0.02em;
    }

    .hero {
        padding: 1.6rem 0 1.2rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.7rem;
        line-height: 1.1;
        font-weight: 800;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-muted);
        max-width: 780px;
    }

    .summary-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 8px 22px rgba(31, 58, 95, 0.05);
        margin-bottom: 1rem;
    }

    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #eef4fb 100%);
        border: 1px solid #d8e3ef;
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: 0 10px 28px rgba(31, 58, 95, 0.08);
        margin-top: 1rem;
    }

    .result-label {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }

    .result-price {
        color: var(--primary);
        font-size: 2.8rem;
        font-weight: 850;
        line-height: 1.05;
        margin-bottom: 0.6rem;
    }

    .result-band {
        color: var(--text-main);
        font-size: 1rem;
        font-weight: 650;
        margin-bottom: 0.65rem;
    }

    .note-box {
        background: #f9fafb;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }

    .stButton > button {
        background: var(--primary);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #2c5282;
        color: white;
        border: none;
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary);
        font-weight: 800;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model_bundle():
    model_path = Path("xgboost_price_model.joblib")

    if not model_path.exists():
        st.error(
            "Model file not found. Make sure xgboost_price_model.joblib is in the same folder as app.py."
        )
        st.stop()

    return joblib.load(model_path)


@st.cache_data
def load_district_options():
    possible_data_paths = [
        Path("residential_single_family_week7_ready.csv"),
        Path("residential_single_family_geocoding_week6.csv"),
        Path("residential_single_family_enriched_week6.csv")
    ]

    for path in possible_data_paths:
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)

                if "DistrictName_Grouped" in df.columns:
                    districts = df["DistrictName_Grouped"].dropna().astype(str).unique()
                elif "DistrictName" in df.columns:
                    districts = df["DistrictName"].dropna().astype(str).unique()
                else:
                    continue

                districts = sorted(districts)

                if "Unknown" not in districts:
                    districts = ["Unknown"] + districts

                return districts

            except Exception:
                continue

    return ["Unknown"]


def get_price_band(price):
    if price < 500_000:
        return "Under $500K"
    if price < 1_000_000:
        return "$500K–$1M"
    if price < 2_000_000:
        return "$1M–$2M"
    return "$2M+"


def get_band_note(price_band):
    if price_band in ["$500K–$1M", "$1M–$2M"]:
        return (
            "This prediction falls within one of the model's strongest evaluation ranges. "
            "In the Week 8 evaluation, the $500K–$2M bands represented most of the test set and had the lowest typical percentage errors."
        )

    if price_band == "Under $500K":
        return (
            "This prediction falls in the lowest price band, where the model was less reliable. "
            "Lower-priced records may include unusual or nonstandard transactions, and percentage errors become larger when actual prices are small."
        )

    return (
        "This prediction falls in the high-end price band, where the model was less reliable. "
        "Luxury properties often depend on features not fully captured by the current inputs, such as views, renovations, architecture, privacy, and micro-location."
    )


def validate_inputs(living_area, bedrooms, bathrooms, lot_size, year_built):
    warnings = []

    if living_area < 500:
        warnings.append("Living area is unusually small for a single-family residence.")
    if living_area > 8_000:
        warnings.append("Living area is unusually large, so the model may be extrapolating.")
    if bedrooms == 0:
        warnings.append("Bedroom count is 0, which may indicate an unusual property record.")
    if bathrooms == 0:
        warnings.append("Bathroom count is 0, so bed/bath ratio cannot be computed normally.")
    if lot_size < 1_000:
        warnings.append("Lot size is unusually small for a single-family residence.")
    if lot_size > 100_000:
        warnings.append("Lot size is unusually large, so the model may be extrapolating.")
    if year_built < 1900:
        warnings.append("The home is very old, so condition and renovation status may matter more than the model can capture.")

    return warnings


def build_input_dataframe(
    features,
    living_area,
    bedrooms,
    bathrooms,
    lot_size,
    district_name,
    year_built,
    prediction_year
):
    input_data = {
        "LivingArea": living_area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "LotSize": lot_size,
        "LivingArea_missing": 0,
        "Bedrooms_missing": 0,
        "Bathrooms_missing": 0,
        "LotSize_missing": 0,
        "BedBathRatio": bedrooms / bathrooms if bathrooms != 0 else np.nan,
        "TotalRoomsProxy": bedrooms + bathrooms,
        "log_LivingArea": np.log1p(max(living_area, 0)),
        "log_LotSize": np.log1p(max(lot_size, 0)),
        "DistrictName": district_name,
        "DistrictName_Grouped": district_name,
        "HasUnifiedDistrict": int(district_name != "Unknown"),
        "YearBuilt": year_built,
        "PropertyAge": prediction_year - year_built
    }

    input_df = pd.DataFrame([input_data])

    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = np.nan

    return input_df[features], input_df


model_bundle = load_model_bundle()
model = model_bundle["model"]
features = model_bundle["features"]
district_options = load_district_options()


with st.sidebar:
    st.header("Model Overview")
    st.write("Model: Week 7 XGBoost")
    st.write("Target: ClosePrice")
    st.write("Test month: June 2026")

    st.divider()

    st.subheader("Overall Test Metrics")
    st.write("R²: 0.615")
    st.write("MAPE: 36.91%")
    st.write("MdAPE: 19.96%")

    st.divider()

    st.subheader("Most Reliable Price Range")
    st.write("$500K–$2M")
    st.caption(
        "The model performed best in the middle price bands, which also represented most of the test set."
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">California Home Price Predictor</div>
        <div class="hero-subtitle">
            Estimate the closing price of a California single-family residence using a trained XGBoost model with engineered property and location features.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


input_col, summary_col = st.columns([1.15, 0.85], gap="large")


with input_col:
    with st.container(border=True):
        st.subheader("Property Details")

        detail_col_1, detail_col_2 = st.columns(2)

        with detail_col_1:
            living_area = st.number_input(
                "Living area, square feet",
                min_value=100,
                max_value=20_000,
                value=1800,
                step=100
            )

            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=20,
                value=3,
                step=1
            )

        with detail_col_2:
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0.0,
                max_value=20.0,
                value=2.0,
                step=0.5
            )

            lot_size = st.number_input(
                "Lot size, square feet",
                min_value=0,
                max_value=1_000_000,
                value=6000,
                step=500
            )

    with st.container(border=True):
        st.subheader("Location and Age")

        district_name = st.selectbox(
            "School district group",
            options=district_options,
            index=district_options.index("Unknown") if "Unknown" in district_options else 0,
            help="Choose Unknown if the school district is unavailable."
        )

        age_col_1, age_col_2 = st.columns(2)

        with age_col_1:
            year_built = st.number_input(
                "Year built",
                min_value=1800,
                max_value=2026,
                value=1980,
                step=1
            )

        with age_col_2:
            prediction_year = st.number_input(
                "Prediction year",
                min_value=2022,
                max_value=2030,
                value=2026,
                step=1
            )


with summary_col:
    with st.container(border=True):
        st.subheader("Derived Features")

        bed_bath_ratio = bedrooms / bathrooms if bathrooms != 0 else np.nan
        total_rooms_proxy = bedrooms + bathrooms
        property_age = prediction_year - year_built
        district_known = district_name != "Unknown"

        metric_col_1, metric_col_2 = st.columns(2)

        with metric_col_1:
            st.metric(
                "Bed/Bath Ratio",
                "N/A" if pd.isna(bed_bath_ratio) else f"{bed_bath_ratio:.2f}"
            )
            st.metric("Property Age", f"{property_age} years")

        with metric_col_2:
            st.metric("Rooms Proxy", f"{total_rooms_proxy:.1f}")
            st.metric("District Known", "Yes" if district_known else "No")

        st.caption("These features are generated automatically before prediction.")

    input_warnings = validate_inputs(
        living_area,
        bedrooms,
        bathrooms,
        lot_size,
        year_built
    )

    if input_warnings:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("Input notes")
        for warning in input_warnings:
            st.write(f"- {warning}")
        st.markdown("</div>", unsafe_allow_html=True)


model_input_df, full_input_df = build_input_dataframe(
    features=features,
    living_area=living_area,
    bedrooms=bedrooms,
    bathrooms=bathrooms,
    lot_size=lot_size,
    district_name=district_name,
    year_built=year_built,
    prediction_year=prediction_year
)


st.markdown("")

predict_clicked = st.button(
    "Predict close price",
    type="primary",
    use_container_width=True
)


if predict_clicked:
    prediction = model.predict(model_input_df)[0]
    prediction = max(prediction, 0)

    predicted_band = get_price_band(prediction)
    band_note = get_band_note(predicted_band)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Close Price</div>
            <div class="result-price">${prediction:,.0f}</div>
            <div class="result-band">Predicted price band: {predicted_band}</div>
            <div class="note-box">{band_note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    detail_tab, limitations_tab = st.tabs(["Model input", "Limitations"])

    with detail_tab:
        st.dataframe(model_input_df, use_container_width=True)

    with limitations_tab:
        st.write(
            "This prediction is an estimate from a machine learning model, not a formal appraisal."
        )
        st.write(
            "The model uses structured property fields and school district information, but it does not fully capture interior condition, renovations, views, street-level location, architecture, seller concessions, unusual sale circumstances, or luxury amenities."
        )
        st.write(
            "Week 8 evaluation showed that the model performs best in the $500K–$2M range and is less reliable at the very low and very high ends of the market."
        )
else:
    st.info("Enter property details, then click Predict close price.")