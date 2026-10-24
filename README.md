# Cloud-Native Real Estate Price Prediction & Telemetry Engine

An end-to-end, production-ready Machine Learning serving microservice built with **FastAPI** and an interactive **Streamlit** dashboard, architected for deployment on **AWS** with zero-cost local simulation capabilities.

🔗 **Live Interactive Demo:** [monam-property-valuation.streamlit.app](https://monam-property-valuation.streamlit.app/)

---

## 🏗️ Architecture & Tech Stack

- **API Framework:** FastAPI, Pydantic (v2), Uvicorn
- **User Interface:** Streamlit, Requests
- **Machine Learning:** Scikit-Learn, Joblib, NumPy
- **Database & ORM:** SQLAlchemy (Dual engine: SQLite for local execution, PostgreSQL / AWS RDS for production)
- **Cloud Infrastructure:** AWS EC2 (Compute), AWS S3 (Model Artifact Storage)
- **Containerization:** Docker

---

## ✨ Core Features

- **Live Web Interface:** Hosted publicly on Streamlit Cloud for real-time model appraisal demos and log monitoring ([Live App](https://monam-property-valuation.streamlit.app/)).
- **Dual-Mode Artifact Loading (Local / AWS S3):** 
  - *Cloud:* Securely streams `model.joblib` from an Amazon S3 bucket on startup via `boto3` and IAM Instance Profiles.
  - *Local:* Gracefully falls back to local disk storage (`model.joblib`) if S3 configurations or bucket credentials are absent, enabling zero-cloud-cost testing.
- **Relational Telemetry & Audit Trails:**
  - Every inference payload, predicted value, and round-trip execution latency (ms) is tracked via SQLAlchemy ORM.
  - Dynamically routes to an embedded **SQLite** instance (`ml_logs.db`) in development, or connects seamlessly to **PostgreSQL (AWS RDS)** in production via the `DATABASE_URL` environment variable.
- **Interactive Streamlit UI (`app.py`):**
  - Intuitive slider and dropdown controls to parameterize property features (Square Footage, Bedrooms, House Age).
  - Live appraisal cards reporting estimated valuation and latency benchmarks.
  - Telemetry console providing direct visibility into historical prediction logs fetched from the API.
- **Container-Ready Architecture:** Includes a streamlined `Dockerfile` configured to pre-train or pull artifacts, install locked dependencies, and serve production traffic over Uvicorn.

---

## 📂 Repository Structure

```text
├── Dockerfile                  # Container blueprint for isolated cloud runtime
├── requirements.txt            # Python dependencies (FastAPI, Streamlit, Scikit-Learn, SQLAlchemy, boto3)
├── main.py                     # FastAPI service, lifespan initialization, and SQLAlchemy schema
├── train_model.py              # Scikit-Learn regression training pipeline & serialization
├── app.py                      # Interactive Streamlit frontend dashboard
├── model.joblib                # Serialized model artifact (generated via train_model.py)
└── sql/
    ├── 01_schema.sql           # PostgreSQL DDL for production tables and indexing
    └── 02_analytics_queries.sql# Analytical queries (Window functions, percentiles, latency tracking)
