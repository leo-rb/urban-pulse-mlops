from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import os

app = FastAPI(title="Urban Pulse API", version="1.0")

# --- CONFIGURATION ---
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "UrbanPulse_Pollution_Prediction"

# Configuration S3 explicite
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "admin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "password")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# On rend le modèle d'entrée flexible pour accepter 'traffic_density' ou 'traffic_volume'
class PredictionInput(BaseModel):
    temperature: float
    humidity: float
    traffic_volume: float = None
    traffic_density: float = None # Pour compatibilité
    # On accepte les autres champs pour ne pas planter, mais on ne les utilisera pas
    hour: int = None
    location_id: int = None

def get_latest_model_uri():
    """Récupère dynamiquement l'URI du dernier modèle entraîné avec succès."""
    print(f"🔌 Connexion à MLflow: {MLFLOW_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise Exception(f"L'expérience '{EXPERIMENT_NAME}' n'existe pas encore.")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise Exception("Aucun run d'entraînement trouvé.")
    
    last_run_id = runs[0].info.run_id
    print(f"✅ Dernier Run ID trouvé : {last_run_id}")
    return f"runs:/{last_run_id}/random_forest_model"

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = get_latest_model_uri()
        model = mlflow.pyfunc.load_model(model_uri)
        print("🎉 Modèle chargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur au chargement du modèle : {e}")

@app.post("/predict")
def predict(input_data: PredictionInput):
    global model
    if model is None:
        try:
            load_model()
        except:
            raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        # --- ETAPE CRUCIALE : NETTOYAGE DES DONNÉES ---
        # 1. On gère l'ambiguïté Volume/Density
        vol = input_data.traffic_volume
        if vol is None and input_data.traffic_density is not None:
            vol = input_data.traffic_density
        if vol is None:
            vol = 0 # Valeur par défaut si tout manque
            
        # 2. On reconstruit un DataFrame PROPRE avec UNIQUEMENT ce que le modèle veut
        # L'ordre des colonnes doit être respecté : ['temperature', 'humidity', 'traffic_volume']
        features = {
            "temperature": input_data.temperature,
            "humidity": input_data.humidity,
            "traffic_volume": vol
        }
        
        df = pd.DataFrame([features])
        
        # Prédiction
        prediction = model.predict(df)
        return {"pollution_prediction": float(prediction[0])}
    
    except Exception as e:
        print(f"🔥 Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}