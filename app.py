import os
import requests
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Real Estate Property Valuation",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend service configuration
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Sidebar Configuration
st.sidebar.title("⚙️ Service Configuration")
backend_url = st.sidebar.text_input("Backend API URL", value=DEFAULT_BACKEND_URL)

# Check backend health
backend_online = False
service_info = {}
try:
    health_res = requests.get(f"{backend_url.rstrip('/')}/", timeout=3)
    if health_res.status_code == 200:
        backend_online = True
        service_info = health_res.json()
        st.sidebar.success("● Backend Connected")
        st.sidebar.caption(
            f"**Model Loaded:** {service_info.get('model_loaded')} | "
            f"**DB:** {service_info.get('database')}"
        )
    else:
        st.sidebar.warning(f"Backend responded with HTTP {health_res.status_code}")
except requests.exceptions.RequestException:
    st.sidebar.error("● Backend Offline (Check FastAPI service)")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Architecture Overview:**
    - **API:** FastAPI + Pydantic v2
    - **ML Model:** Scikit-Learn Linear Regression
    - **Persistence:** SQLite (`ml_logs.db`) / AWS RDS
    - **Storage:** Local Artifact / AWS S3
    """
)

# Header Section
st.title("🏡 Real Estate Property Valuation Microservice")
st.markdown(
    "Estimate residential property market value in real-time with automated audit logging "
    "and low-latency inference telemetry."
)
st.markdown("---")

# Layout: Split into Left (Inputs) and Right (Valuation Output)
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📋 Property Features")
    with st.form("valuation_form"):
        sqft = st.number_input(
            "Square Footage (sqft)",
            min_value=100.0,
            max_value=25000.0,
            value=2200.0,
            step=50.0,
            help="Total living area in square feet (must be > 0)."
        )
        
        bedrooms = st.slider(
            "Bedrooms",
            min_value=0,
            max_value=12,
            value=3,
            step=1,
            help="Total number of bedrooms in the property."
        )
        
        age = st.slider(
            "Property Age (Years)",
            min_value=0,
            max_value=120,
            value=10,
            step=1,
            help="Age of the property structure in years."
        )

        submit_btn = st.form_submit_button(
            "Appraise Property",
            use_container_width=True,
            type="primary"
        )

# Session state initialization for latest prediction
if "latest_prediction" not in st.session_state:
    st.session_state["latest_prediction"] = None

# Handle Form Submission
if submit_btn:
    payload = {
        "sqft": float(sqft),
        "bedrooms": int(bedrooms),
        "age": int(age)
    }
    
    try:
        with st.spinner("Requesting valuation from FastAPI backend..."):
            response = requests.post(
                f"{backend_url.rstrip('/')}/predict",
                json=payload,
                timeout=5
            )
        
        if response.status_code == 200:
            st.session_state["latest_prediction"] = response.json()
            st.toast("Valuation appraisal complete!", icon="✅")
        elif response.status_code == 422:
            st.error(f"Input Validation Error: {response.json().get('detail')}")
        elif response.status_code == 503:
            st.error("Model Not Ready: Model artifact is missing on server.")
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend at {backend_url}. Details: {e}")

# Right Column: Metric Card Results
with right_col:
    st.subheader("📊 Appraisal Results")
    pred_data = st.session_state.get("latest_prediction")

    if pred_data and pred_data.get("status") == "success":
        price = pred_data.get("prediction", pred_data.get("predicted_price", 0.0))
        latency = pred_data.get("latency_ms", 0.0)
        log_id = pred_data.get("log_id", "N/A")

        # Top Highlight Card
        st.success("Valuation computed successfully!")

        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric(
                label="Estimated Price",
                value=f"${price:,.2f}"
            )
        with col_metric2:
            st.metric(
                label="Inference Latency",
                value=f"{latency:.2f} ms"
            )
        with col_metric3:
            st.metric(
                label="Audit Log ID",
                value=f"#{log_id}"
            )

        st.caption(
            f"**Evaluated Specs:** {pred_data.get('features', {}).get('sqft')} sqft | "
            f"{pred_data.get('features', {}).get('bedrooms')} beds | "
            f"{pred_data.get('features', {}).get('age')} yrs old"
        )
    else:
        st.info("👈 Enter property features on the left and click **Appraise Property** to view valuation.")

# Bottom Section: Telemetry & Audit Logs
st.markdown("---")
st.subheader("📡 Telemetry & Recent Prediction Audit Logs")

col_table_header, col_refresh = st.columns([5, 1])
with col_table_header:
    st.caption("Displaying the 10 most recent predictions saved to the database (`GET /logs`).")
with col_refresh:
    refresh_logs = st.button("🔄 Refresh Logs", use_container_width=True)

try:
    logs_res = requests.get(f"{backend_url.rstrip('/')}/logs", timeout=3)
    if logs_res.status_code == 200:
        logs_data = logs_res.json()
        if logs_data:
            df = pd.DataFrame(logs_data)
            # Reorder and format columns for readability
            col_order = ["id", "sqft", "bedrooms", "age", "predicted_price", "latency_ms", "timestamp"]
            df = df[[c for c in col_order if c in df.columns]]
            df["predicted_price"] = df["predicted_price"].apply(lambda p: f"${p:,.2f}" if pd.notnull(p) else "$0.00")
            df["latency_ms"] = df["latency_ms"].apply(lambda l: f"{l:.2f} ms" if pd.notnull(l) else "0.00 ms")
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.write("No predictions logged yet. Run an appraisal above to generate records.")
    else:
        st.warning(f"Could not load audit logs. Backend responded with status {logs_res.status_code}.")
except requests.exceptions.RequestException:
    st.warning("Backend service unavailable. Start FastAPI backend to view telemetry logs.")
