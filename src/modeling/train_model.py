import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
print("🔧 Configuration de l'environnement...")

DB_USER = os.getenv("POSTGRES_USER", "urban_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "urban_password")
DB_NAME = os.getenv("POSTGRES_DB", "urban_pulse_db")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5433")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def load_data():
    print("⏳ Chargement des données depuis la DB...")
    connection_str = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_str)
    
    query = "SELECT * FROM sensor_data"
    df = pd.read_sql(query, engine)
    return df

def train():
    print(f"🚀 Démarrage de l'entraînement vers {TRACKING_URI}...")
    
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("UrbanPulse_Pollution_Prediction")
    
    try:
        df = load_data()
    except Exception as e:
        print(f"❌ Erreur de connexion DB: {e}")
        return

    if df.empty:
        print("⚠️ Pas assez de données !")
        return

    print(f"✅ Données chargées : {len(df)} lignes.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['hour'] = df['timestamp'].dt.hour
    
    # Préparation
    X = df[['traffic_density', 'temperature', 'humidity', 'hour', 'location_id']]
    y = df['co2_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        print("🤖 Entraînement du modèle...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"📉 MAE: {mae}")

        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print("🎉 Modèle entraîné et sauvegardé avec succès dans MLflow !")

if __name__ == "__main__":
    train()