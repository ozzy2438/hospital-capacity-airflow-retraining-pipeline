# hospital-capacity-airflow-retraining-pipeline

Apache Airflow project that implements **scheduled (monthly) model retraining**, **champion vs challenger evaluation**, and **gated auto-promotion** for a hospital capacity forecasting/classification model.

## What this does

This DAG:

- Retrains a candidate model on a monthly schedule
- Evaluates candidate vs current production model
- Runs basic drift checks
- Only promotes the candidate if the configured criteria pass

## Quickstart

This repository is designed to be dropped into an Airflow deployment (Docker, MWAA, Composer, Astronomer, etc.).

### Environment variables

- `MODEL_DIR` (default `/opt/airflow/models`)
- `DATA_DB` (default `/opt/airflow/data/hospital_capacity.db`)
- `AUC_IMPROVEMENT_THRESHOLD` (default `0.01`)
- `RECALL_REGRESSION_THRESHOLD` (default `0.10`)
- `DRIFT_KS_THRESHOLD` (default `0.01`)
- `DRIFT_PSI_THRESHOLD` (default `0.25`)

### Files

- `dags/model_retraining_dag.py` is the Airflow DAG

## Notes

This is a reference implementation. For production, prefer storing artifacts in object storage or a model registry and use a robust promotion mechanism (registry stage, pointer update, etc.).
