import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path

# --- CONFIGURATION ---
# Chargement du .env pour la DB
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Connexion DB (Port 5433)
DB_USER = os.getenv("POSTGRES_USER", "urban_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "urban_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "urban_pulse_db")

# URL de ton API (FastAPI)
default_api_url = "http://localhost:8000/predict"
API_URL = os.getenv("API_URL", default_api_url)

# Configuration de la page
st.set_page_config(page_title="Urban Pulse Dashboard", layout="wide")

st.title("🏙️ Urban Pulse : AI Control Center")
st.markdown("Surveillance de la pollution et prédictions IA en temps réel.")

# --- PARTIE 1 : VISUALISATION (Données Historiques) ---
st.header("1. Données en Temps Réel (Postgres)")

def load_data():
    """Charge les 100 dernières mesures depuis la base de données"""
    try:
        connection_str = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_str)
        query = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 100"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données : {e}")
        return pd.DataFrame()

if st.button('🔄 Rafraîchir les données'):
    st.rerun()

df = load_data()

if not df.empty:
    # Création de deux colonnes pour les graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Niveau de CO2")
        fig_co2 = px.line(df, x='timestamp', y='co2_level', title='Évolution du CO2', color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_co2, use_container_width=True)
        
    with col2:
        st.subheader("🚗 Densité du Trafic")
        fig_traffic = px.area(df, x='timestamp', y='traffic_density', title='Densité du Trafic', color_discrete_sequence=['#3d9dfc'])
        st.plotly_chart(fig_traffic, use_container_width=True)

    # Affichage des statistiques
    st.metric("Dernier relevé CO2", f"{df.iloc[0]['co2_level']} ppm", delta_color="inverse")
else:
    st.warning("Aucune donnée trouvée. Vérifie que le générateur (mock_generator.py) tourne bien.")

st.markdown("---")

# --- PARTIE 2 : SIMULATION IA (Appel API) ---
st.header("2. Simulateur Prédictif (IA)")
st.info("Utilisez les curseurs ci-dessous pour simuler une situation et demander une prédiction à l'IA.")

# Formulaire de saisie (Sidebar ou Colonnes)
c1, c2, c3 = st.columns(3)

with c1:
    traffic = st.slider("🚗 Densité Trafic", 0, 100, 50)
    hour = st.slider("🕒 Heure de la journée", 0, 23, 12)

with c2:
    temp = st.slider("🌡️ Température (°C)", -10.0, 40.0, 20.0)
    humidity = st.slider("💧 Humidité (%)", 0, 100, 50)

with c3:
    location = st.selectbox("📍 Quartier (ID)", [1, 2, 3])
    predict_btn = st.button("🔮 Lancer la Prédiction", type="primary")

# Action du bouton
if predict_btn:
    # Préparation du JSON pour l'API
    payload = {
        "traffic_density": traffic,
        "temperature": temp,
        "humidity": humidity,
        "hour": hour,
        "location_id": location
    }
    
    try:
        # Appel à l'API
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            pred_co2 = result['predicted_co2']
            alert = result['alert_level']
            
            st.success("✅ Prédiction réussie !")
            
            # Affichage du résultat en gros
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Pollution Prédite (CO2)", f"{pred_co2} ppm")
            
            if alert == "HIGH":
                col_res2.error("🚨 ALERTE : NIVEAU CRITIQUE")
            else:
                col_res2.success("🟢 NIVEAU NORMAL")
                
        else:
            st.error(f"Erreur API : {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de contacter l'API. Vérifie que 'api.py' est bien lancé dans un terminal.")