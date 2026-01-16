import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
st.set_page_config(page_title="Urban Pulse Dashboard", layout="wide")

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "urban_pulse_db")
DB_USER = os.getenv("DB_USER", "urban_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "urban_password")

# Connexion Base de données
@st.cache_resource
def get_db_engine():
    conn_str = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"
    return create_engine(conn_str)

engine = get_db_engine()

# --- CHARGEMENT DES DONNÉES ---
def load_data():
    query = "SELECT * FROM pollution_data ORDER BY timestamp DESC LIMIT 100"
    return pd.read_sql(query, engine)

st.title("🏙️ Urban Pulse - Monitoring Air & Trafic")

try:
    df = load_data()

    # --- LAYOUT DASHBOARD ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Niveau de Pollution (AQI)")
        fig_line = px.line(df, x='timestamp', y='pollution_level', 
                           color_discrete_sequence=['#FF4B4B'], markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.subheader("Volume du Trafic")
        fig_traffic = px.area(df, x='timestamp', y='traffic_volume', 
                             color_discrete_sequence=['#3d9dfc'])
        st.plotly_chart(fig_traffic, use_container_width=True)

    st.subheader("Corrélation Trafic vs Pollution")
    fig_scatter = px.scatter(df, x='traffic_volume', y='pollution_level', color='temperature',
                             size='humidity', hover_data=['timestamp'])
    st.plotly_chart(fig_scatter, use_container_width=True)

except Exception as e:
    st.error(f"Erreur de connexion à la base de données : {e}")

# --- SIDEBAR : PRÉDICTION ---
st.sidebar.header("🔮 Prédiction IA")
temp = st.sidebar.slider("Température (°C)", -10, 45, 20)
hum = st.sidebar.slider("Humidité (%)", 0, 100, 50)
traf = st.sidebar.number_input("Volume Trafic (véhicules/h)", 0, 5000, 500)

if st.sidebar.button("Lancer la prédiction 🚀"):
    payload = {
        "temperature": float(temp),
        "humidity": float(hum),
        "traffic_volume": float(traf)
    }
    
    try:
        endpoint = f"{API_URL}/predict"
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # Récupération de la valeur (gestion des deux clés possibles)
            pred_value = result.get("pollution_prediction", result.get("predicted_co2", 0))
            
            st.sidebar.success(f"🎯 Pollution estimée : {pred_value:.2f}")
            
            # Affichage de l'alerte locale
            if pred_value > 100:
                st.sidebar.error("🚨 Alerte : Pollution élevée !")
            elif pred_value > 50:
                st.sidebar.warning("⚠️ Attention : Qualité moyenne")
            else:
                st.sidebar.info("✅ Qualité de l'air : Bonne")
        else:
            st.sidebar.error(f"Erreur API ({response.status_code}) : {response.text}")
            
    except Exception as e:
        st.sidebar.error(f"Impossible de contacter l'API : {e}")