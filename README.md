# Cloud-Native Real Estate Price Prediction API

An end-to-end, production-grade Machine Learning inference API built with **FastAPI** and deployed on **AWS**. 

This project demonstrates a complete MLOps serving architecture. It transitions a trained Scikit-Learn regression model from a local environment to a containerized, cloud-native API capable of high-throughput inference, artifact management, and relational telemetry logging.

## 🏗️ Architecture & Tech Stack

* **API Framework:** FastAPI, Pydantic, Uvicorn
* **Containerization:** Docker
* **Cloud Infrastructure:** AWS EC2 (Compute), AWS S3 (Model Artifact Storage)
* **Database & ORM:** PostgreSQL (AWS RDS), SQLAlchemy
* **Machine Learning:** Scikit-Learn, Joblib, NumPy

## ✨ Core Features

* **Dynamic Artifact Loading (AWS S3):** The API does not hardcode heavy model weights into the Docker image. Instead, it securely downloads the `model.joblib` artifact from an Amazon S3 bucket at startup using `boto3` and EC2 IAM Instance Profiles.
* **Relational Telemetry Logging (AWS RDS):** Every inference request, alongside its execution latency and predicted output, is asynchronously logged to a PostgreSQL database via SQLAlchemy ORM for performance monitoring and audit trailing.
* **Advanced Analytics Engine:** Includes custom DDL schemas and raw SQL queries utilizing window functions and aggregations to track P95 latency and rolling prediction averages over time.
* **Containerized Deployment:** Fully isolated environment using Docker, ensuring 100% parity between local development and cloud production.

## 📂 Repository Structure

```text
├── Dockerfile                  # Container blueprint
├── requirements.txt            # Python dependencies
├── main.py                     # FastAPI application and SQLAlchemy ORM logic
├── train_model.py              # Scikit-Learn model training and serialization script
└── sql/
    ├── 01_schema.sql           # PostgreSQL DDL for production tables and indexing
    └── 02_analytics_queries.sql# Advanced analytical queries (Window functions, percentiles)
