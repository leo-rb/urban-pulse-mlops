import time
import random
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from pathlib import Path

# --- 1. CHARGEMENT ROBUSTE DU .ENV ---
# On cherche le fichier .env en remontant les dossiers
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- 2. RECUPERATION DES VARIABLES ---
DB_USER = os.getenv("POSTGRES_USER", "urban_user") # Valeur par défaut si échec
DB_PASS = os.getenv("POSTGRES_PASSWORD", "urban_password")
DB_NAME = os.getenv("POSTGRES_DB", "urban_pulse_db")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")

print(f"🔧 DEBUG: Tentative de connexion vers {DB_HOST}...")
print(f"🔧 DEBUG: Utilisateur={DB_USER}, DB={DB_NAME}")

# --- 3. CREATION DU MOTEUR ---
try:
    connection_string = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string)
    
    # Test immédiat de connexion
    with engine.connect() as conn:
        print("✅ CONNEXION RÉUSSIE À LA BASE DE DONNÉES !")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE DE CONNEXION : {e}")
    exit()

# --- 4. LA SIMULATION ---
LOCATIONS = [
    {"id": 1, "name": "Centre-Ville", "lat": 48.8566, "lon": 2.3522},
    {"id": 2, "name": "Zone Industrielle", "lat": 48.8600, "lon": 2.3600},
    {"id": 3, "name": "Parc Floral", "lat": 48.8400, "lon": 2.4500},
    {"id": 4, "name": "Périphérique Nord", "lat": 48.9000, "lon": 2.3500}
]

def generate_sensor_data():
    data = []
    current_time = datetime.now()
    
    for loc in LOCATIONS:
        traffic_factor = random.uniform(0.5, 1.5)
        pollution_factor = random.uniform(0.5, 1.5)

        if loc['name'] == "Périphérique Nord":
            traffic = random.randint(50, 120)
            co2 = random.uniform(400, 800) * pollution_factor
        elif loc['name'] == "Parc Floral":
            traffic = random.randint(0, 10)
            co2 = random.uniform(300, 400)
        else:
            traffic = random.randint(10, 60) * traffic_factor
            co2 = random.uniform(350, 500) * pollution_factor

        record = {
            "timestamp": current_time,
            "location_id": loc['id'],
            "location_name": loc['name'],
            "temperature": round(random.uniform(10, 25), 1),
            "traffic_density": int(traffic),
            "co2_level": round(co2, 2),
            "humidity": round(random.uniform(30, 70), 1)
        }
        data.append(record)
    return data

def run_simulation():
    print("🚀 Démarrage de la simulation des capteurs...")
    
    try:
        while True:
            sensor_data = generate_sensor_data()
            df = pd.DataFrame(sensor_data)
            
           
            df.to_sql('sensor_data', engine, if_exists='append', index=False)
            
            print(f"📡 {len(df)} mesures envoyées à {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation arrêtée.")

if __name__ == "__main__":
    run_simulation()