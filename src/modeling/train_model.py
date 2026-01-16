import os
import time
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine
import boto3
from botocore.exceptions import ClientError

# --- CONFIGURATION (Match avec docker-compose.yml) ---
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "urban_pulse_db")
DB_USER = os.getenv("POSTGRES_USER", "urban_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "urban_password")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "admin")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "password")

def ensure_bucket_exists(bucket_name="mlflow"):
    """
    Vérifie si le bucket S3 existe, sinon le crée.
    C'est l'assurance vie contre l'erreur 'NoSuchBucket'.
    """
    print(f"🔍 Vérification du bucket S3 '{bucket_name}' sur {S3_ENDPOINT}...")
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"✅ Le bucket '{bucket_name}' existe déjà.")
    except ClientError:
        print(f"⚠️ Le bucket '{bucket_name}' n'existe pas. Création en cours...")
        try:
            s3.create_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name