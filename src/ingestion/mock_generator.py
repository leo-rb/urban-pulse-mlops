import time
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "urban_pulse_db")
DB_USER = os.getenv("POSTGRES_USER", "urban_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "urban_password")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_engine():
    """Crée le moteur de connexion avec une boucle de réessai (Retry Logic)."""
    max_retries = 20
    wait_seconds = 5
    
    for i in range(max_retries):
        try:
            print(f"🔌 Tentative de connexion à la DB ({i+1}/{max_retries})...")
            engine = create_engine(DATABASE_URL)
            # Test simple pour vérifier la connexion
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Connexion à la DB réussie !")
            return engine
        except Exception as e:
            print(f"⚠️ La DB n'est pas encore prête : {e}")
            print(f"⏳ Attente de {wait_seconds} secondes...")
            time.sleep(wait_seconds)
    
    raise Exception("❌ Impossible de se connecter à la DB après plusieurs tentatives.")

def generate_mock_data(n_rows=50):
    """Génère des données factices compatibles avec le modèle."""
    data = {
        "timestamp": [datetime.now() - timedelta(minutes=i) for i in range(n_rows)],
        "pollution_level": [random.uniform(10, 150) for _ in range(n_rows)],
        "temperature": [random.uniform(-5, 35) for _ in range(n_rows)],
        "humidity": [random.uniform(20, 90) for _ in range(n_rows)],
        "traffic_volume": [random.randint(50, 2000) for _ in range(n_rows)]
    }
    return pd.DataFrame(data)

def main():
    print("🚀 Démarrage du générateur de données...")
    
    # 1. Obtenir la connexion (avec attente)
    engine = get_engine()
    
    while True:
        # 2. Générer les données
        df = generate_mock_data(n_rows=50)
        
        # 3. Sauvegarder dans la DB
        try:
            print("💾 Sauvegarde des données dans la table 'pollution_data'...")
            df.to_sql('pollution_data', engine, if_exists='append', index=False)
            print(f"✅ Succès ! {len(df)} lignes insérées.")
        except Exception as e:
            print(f"❌ Erreur lors de l'insertion : {e}")
        
        # Pause de 5 secondes avant la prochaine vague
        time.sleep(5)

if __name__ == "__main__":
    main()