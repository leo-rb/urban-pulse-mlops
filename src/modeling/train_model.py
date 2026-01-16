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
            print(f"✅ Bucket '{bucket_name}' créé avec succès !")
        except Exception as e:
            print(f"❌ Impossible de créer le bucket : {e}")
            raise e

def get_data_from_db():
    """
    Récupère les données depuis PostgreSQL avec une boucle de réessai.
    """
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    
    query = "SELECT * FROM pollution_data"
    
    # Tentative de connexion (Retry logic)
    max_retries = 5
    for i in range(max_retries):
        try:
            print("⏳ Chargement des données depuis la DB...")
            df = pd.read_sql(query, engine)
            print(f"✅ Données chargées : {len(df)} lignes.")
            return df
        except Exception as e:
            print(f"⚠️ Tentative {i+1}/{max_retries} échouée : {e}")
            time.sleep(5) # Attendre 5 sec avant de réessayer
            
    raise Exception("❌ Impossible de se connecter à la base de données après plusieurs essais.")

def train():
    # 1. Préparer l'infrastructure (Bucket S3)
    ensure_bucket_exists("mlflow")

    # 2. Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("UrbanPulse_Pollution_Prediction")

    # 3. Charger les données
    df = get_data_from_db()

    # Préparation simple des features (X) et target (y)
    # On suppose que 'pollution_level' est la cible et on garde les colonnes numériques
    target = 'pollution_level'
    features = ['temperature', 'humidity', 'traffic_volume']
    
    # Vérification que les colonnes existent
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        print("❌ Erreur : Aucune feature valide trouvée dans les données.")
        return

    X = df[available_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Lancer l'entraînement MLflow
    print(f"🚀 Démarrage de l'entraînement vers {MLFLOW_URI}...")
    
    with mlflow.start_run():
        # Hyperparamètres
        n_estimators = 100
        max_depth = 10
        
        # Log des paramètres
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Entraînement
        print("🤖 Entraînement du modèle...")
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Evaluation
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"📉 MAE: {mae}")
        
        # Log des métriques
        mlflow.log_metric("mae", mae)

        # Sauvegarde du modèle (C'est ici que le bucket S3 est utilisé)
        print("💾 Sauvegarde du modèle dans MinIO...")
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print("🎉 Modèle entraîné et sauvegardé avec succès dans MLflow !")

if __name__ == "__main__":
    train()