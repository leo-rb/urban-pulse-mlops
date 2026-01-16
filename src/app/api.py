import os
import time
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- CONFIG DOCKER (Valeurs par défaut pour la démo) ---
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("MINIO_ROOT_USER", "admin")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("MINIO_ROOT_PASSWORD", "password")

app = FastAPI(title="Urban Pulse API", description="Démo Automatique")
model = None

def get_latest_model():
    """Cherche automatiquement le dernier modèle entraîné dans MLflow"""
    client = MlflowClient()
    experiment_name = "UrbanPulse_Pollution_Prediction"
    
    # On cherche l'expérience
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print("⚠️ Expérience non trouvée (L'entraînement n'a pas encore commencé ?)")
        return None

    # On cherche le run le plus récent
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        return None
    
    latest_run_id = runs[0].info.run_id
    print(f"🔄 Dernier modèle trouvé : {latest_run_id}")
    return f"runs:/{latest_run_id}/random_forest_model"

@app.on_event("startup")
def load_model():
    # On ne fait rien au démarrage bloquant, on chargera à la première requête
    pass

class PredictionInput(BaseModel):
    traffic_density: float
    temperature: float
    humidity: float
    hour: int
    location_id: int

@app.post("/predict")
def predict(input_data: PredictionInput):
    global model
    
    # LAZY LOADING : Si le modèle n'est pas là, on essaie de le charger maintenant
    if model is None:
        model_uri = get_latest_model()
        if model_uri:
            try:
                model = mlflow.pyfunc.load_model(model_uri)
                print("✅ Modèle chargé à la volée !")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Erreur chargement S3: {e}")
        else:
            raise HTTPException(status_code=503, detail="Modèle en cours d'entraînement... Réessayez dans 30 secondes !")

    # Prédiction normale
    data = pd.DataFrame([input_data.dict()])
    prediction = model.predict(data)
    alert = "HIGH" if prediction[0] > 450 else "NORMAL"
    
    return {"predicted_co2": round(prediction[0], 2), "alert_level": alert}