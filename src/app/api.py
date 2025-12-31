import os
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Charger les variables d'environnement (pour le local)
load_dotenv()

# --- CONFIGURATION DYNAMIQUE (DOCKER vs LOCAL) ---

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")


os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("MINIO_ROOT_USER", "admin")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("MINIO_ROOT_PASSWORD", "password")

# --- TON RUN ID (À mettre à jour manuellement après chaque entraînement)
RUN_ID = "17ecca6742ee404caff6fd0454fcaf19"  
app = FastAPI(title="Urban Pulse API", description="Prédiction de pollution CO2")

# Variable globale pour stocker le modèle chargé
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = f"runs:/{RUN_ID}/random_forest_model"
        print(f"📥 Chargement du modèle depuis : {model_uri} ...")
        print(f"🌍 Endpoint S3 utilisé : {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur critique : Impossible de charger le modèle. {e}")
   

class PredictionInput(BaseModel):
    traffic_density: float
    temperature: float
    humidity: float
    hour: int
    location_id: int

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")
    
    try:
        # Conversion en DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Prédiction
        prediction = model.predict(data)
        
        # Seuil d'alerte (Calibré pour la démo)
        alert_level = "HIGH" if prediction[0] > 420 else "NORMAL"
        
        return {
            "predicted_co2": round(float(prediction[0]), 2),
            "alert_level": alert_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")