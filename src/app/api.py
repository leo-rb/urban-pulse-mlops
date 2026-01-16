from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import os

app = FastAPI(title="Urban Pulse API")

# --- CONFIGURATION ---
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "UrbanPulse_Pollution_Prediction"

# Configuration S3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "admin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "password")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

class PredictionInput(BaseModel):
    temperature: float
    humidity: float
    traffic_volume: float = None
    traffic_density: float = None

def get_latest_model_uri():
    print(f"🔌 Connexion à MLflow: {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise Exception("Expérience introuvable")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise Exception("Aucun run trouvé")
    return f"runs:/{runs[0].info.run_id}/random_forest_model"

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(get_latest_model_uri())
        print("✅ Modèle chargé !")
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")

@app.post("/predict")
def predict(input_data: PredictionInput):
    global model
    if not model:
        try:
            load_model()
        except:
            raise HTTPException(status_code=503, detail="Modèle HS")
    
    try:
        # Nettoyage et préparation (Gestion Volume vs Density)
        vol = input_data.traffic_volume
        if vol is None:
            vol = input_data.traffic_density
        if vol is None:
            vol = 0
        
        # DataFrame avec les colonnes exactes attendues par le modèle
        df = pd.DataFrame([{
            "temperature": input_data.temperature,
            "humidity": input_data.humidity,
            "traffic_volume": vol
        }])
        
        pred = model.predict(df)
        return {"pollution_prediction": float(pred[0])}
    except Exception as e:
        print(f"🔥 Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}