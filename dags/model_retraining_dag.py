from __future__ import annotations

"""
Hospital Capacity Model Retraining DAG

What this DAG does
- Retrains on a monthly schedule
- Evaluates candidate vs current production
- Only promotes the candidate if promotion criteria pass

Notes
- Uses SQLite for demo purposes
- Stores artifacts locally in MODEL_DIR (Airflow workers need shared storage)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

from datetime import timedelta
import json
import os
import shutil
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss


# Configuration
MODEL_DIR = os.getenv('MODEL_DIR', '/opt/airflow/models')
DATA_DB = os.getenv('DATA_DB', '/opt/airflow/data/hospital_capacity.db')

PROD_MODEL_PATH = os.path.join(MODEL_DIR, 'production_model.pkl')
CANDIDATE_MODEL_PATH = os.path.join(MODEL_DIR, 'candidate_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')


# Auto-promotion thresholds
AUC_IMPROVEMENT_THRESHOLD = float(os.getenv('AUC_IMPROVEMENT_THRESHOLD', '0.01'))
RECALL_REGRESSION_THRESHOLD = float(os.getenv('RECALL_REGRESSION_THRESHOLD', '0.10'))
DRIFT_KS_THRESHOLD = float(os.getenv('DRIFT_KS_THRESHOLD', '0.01'))
DRIFT_PSI_THRESHOLD = float(os.getenv('DRIFT_PSI_THRESHOLD', '0.25'))


def _safe_json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f_handle:
        json.dump(obj, f_handle, indent=2, default=str)


def extract_training_data(**context):
    conn = sqlite3.connect(DATA_DB)
    query = "SELECT * FROM hospital_capacity_features WHERE date >= date('now', '-12 months') ORDER BY date"
    df_data = pd.read_sql_query(query, conn)
    conn.close()

    os.makedirs(MODEL_DIR, exist_ok=True)
    train_path = os.path.join(MODEL_DIR, 'train.csv')
    df_data.to_csv(train_path, index=False)
    context['ti'].xcom_push(key='train_path', value=train_path)


def train_candidate_model(**context):
    train_path = context['ti'].xcom_pull(key='train_path')
    df_data = pd.read_csv(train_path)

    y = df_data['target'].values
    X = df_data.drop(columns=['target', 'date'], errors='ignore')

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, CANDIDATE_MODEL_PATH)


def _load_model_or_none(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def evaluate_models(**context):
    train_path = context['ti'].xcom_pull(key='train_path')
    df_data = pd.read_csv(train_path)

    y = df_data['target'].values
    X = df_data.drop(columns=['target', 'date'], errors='ignore')

    candidate = joblib.load(CANDIDATE_MODEL_PATH)
    production = _load_model_or_none(PROD_MODEL_PATH)

    candidate_proba = candidate.predict_proba(X)[:, 1]
    cand_auc = float(roc_auc_score(y, candidate_proba))
    cand_precision = float(precision_score(y, (candidate_proba >= 0.5).astype(int), zero_division=0))
    cand_recall = float(recall_score(y, (candidate_proba >= 0.5).astype(int), zero_division=0))
    cand_brier = float(brier_score_loss(y, candidate_proba))

    prod_auc = None
    prod_precision = None
    prod_recall = None
    prod_brier = None

    if production is not None:
        prod_proba = production.predict_proba(X)[:, 1]
        prod_auc = float(roc_auc_score(y, prod_proba))
        prod_precision = float(precision_score(y, (prod_proba >= 0.5).astype(int), zero_division=0))
        prod_recall = float(recall_score(y, (prod_proba >= 0.5).astype(int), zero_division=0))
        prod_brier = float(brier_score_loss(y, prod_proba))

    metrics = {
        'candidate': {
            'auc': cand_auc,
            'precision': cand_precision,
            'recall': cand_recall,
            'brier': cand_brier,
        },
        'production': {
            'auc': prod_auc,
            'precision': prod_precision,
            'recall': prod_recall,
            'brier': prod_brier,
        },
    }

    _safe_json_dump(metrics, METRICS_PATH)
    context['ti'].xcom_push(key='metrics_path', value=METRICS_PATH)


def _psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def drift_check(**context):
    train_path = context['ti'].xcom_pull(key='train_path')
    df_data = pd.read_csv(train_path)
    X = df_data.drop(columns=['target', 'date'], errors='ignore')

    # Simple drift proxy: compare last 30 days vs prior period feature distributions
    if 'date' not in df_data.columns:
        context['ti'].xcom_push(key='drift_pass', value=True)
        return

    df_data['date'] = pd.to_datetime(df_data['date'])
    max_date = df_data['date'].max()
    recent_mask = df_data['date'] >= (max_date - pd.Timedelta(days=30))

    X_recent = df_data.loc[recent_mask].drop(columns=['target', 'date'], errors='ignore')
    X_hist = df_data.loc[~recent_mask].drop(columns=['target', 'date'], errors='ignore')

    if len(X_recent) < 50 or len(X_hist) < 50:
        context['ti'].xcom_push(key='drift_pass', value=True)
        return

    ks_fail = 0
    psi_fail = 0

    for col in X.columns:
        recent_vals = pd.to_numeric(X_recent[col], errors='coerce').dropna().values
        hist_vals = pd.to_numeric(X_hist[col], errors='coerce').dropna().values
        if len(recent_vals) < 10 or len(hist_vals) < 10:
            continue

        ks_p = stats.ks_2samp(hist_vals, recent_vals).pvalue
        if ks_p < DRIFT_KS_THRESHOLD:
            ks_fail += 1

        psi_val = _psi(hist_vals, recent_vals)
        if psi_val > DRIFT_PSI_THRESHOLD:
            psi_fail += 1

    drift_pass = (ks_fail == 0) and (psi_fail == 0)
    context['ti'].xcom_push(key='drift_pass', value=bool(drift_pass))


def decide_promotion(**context):
    drift_pass = context['ti'].xcom_pull(key='drift_pass')

    with open(METRICS_PATH, 'r') as f_handle:
        metrics = json.load(f_handle)

    cand_auc = metrics['candidate']['auc']
    prod_auc = metrics['production']['auc']
    cand_recall = metrics['candidate']['recall']
    prod_recall = metrics['production']['recall']

    # No production model yet: promote first candidate
    if prod_auc is None:
        return 'promote_candidate'

    auc_improved = cand_auc >= (prod_auc + AUC_IMPROVEMENT_THRESHOLD)
    recall_ok = True
    if prod_recall is not None:
        recall_ok = cand_recall >= (prod_recall - RECALL_REGRESSION_THRESHOLD)

    if bool(drift_pass) and bool(auc_improved) and bool(recall_ok):
        return 'promote_candidate'
    return 'skip_promotion'


def promote_candidate_model(**context):
    os.makedirs(MODEL_DIR, exist_ok=True)
    shutil.copy2(CANDIDATE_MODEL_PATH, PROD_MODEL_PATH)


default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    dag_id='hospital_capacity_model_retraining',
    description='Monthly retraining + evaluation + gated auto-promotion',
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval='@monthly',
    catchup=False,
    tags=['ml', 'retraining', 'hospital-capacity'],
) as dag:

    t_extract = PythonOperator(
        task_id='extract_training_data',
        python_callable=extract_training_data,
        provide_context=True,
    )

    t_train = PythonOperator(
        task_id='train_candidate_model',
        python_callable=train_candidate_model,
        provide_context=True,
    )

    t_eval = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        provide_context=True,
    )

    t_drift = PythonOperator(
        task_id='drift_check',
        python_callable=drift_check,
        provide_context=True,
    )

    t_branch = BranchPythonOperator(
        task_id='decide_promotion',
        python_callable=decide_promotion,
        provide_context=True,
    )

    t_promote = PythonOperator(
        task_id='promote_candidate',
        python_callable=promote_candidate_model,
        provide_context=True,
    )

    t_skip = EmptyOperator(task_id='skip_promotion')

    t_done = EmptyOperator(task_id='done')

    t_extract >> t_train >> t_eval >> t_drift >> t_branch
    t_branch >> t_promote >> t_done
    t_branch >> t_skip >> t_done
