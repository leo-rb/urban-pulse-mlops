# 🌍 Urban Pulse - Prédiction de Pollution CO2 (MLOps)

Ce projet est une solution complète **End-to-End MLOps** permettant de prédire les niveaux de CO2 en fonction de données urbaines (trafic, météo, heure).

L'architecture est entièrement conteneurisée avec **Docker** et suit les meilleures pratiques DevOps.

## 🏗️ Architecture Technique

Le projet est composé de 5 micro-services orchestrés par Docker Compose :

1.  **Ingestion :** Script Python simulant des capteurs IoT en temps réel.
2.  **Stockage :** Base de données **PostgreSQL** pour les données brutes.
3.  **Tracking & Registry :** Serveur **MLflow** couplé à **MinIO** (S3 compatible) pour le stockage des artefacts (modèles).
4.  **Serving :** API REST développée avec **FastAPI** pour exposer le modèle.
5.  **Frontend :** Dashboard interactif **Streamlit** pour les utilisateurs finaux.

## 🚀 Comment lancer le projet ?

Pré-requis : Avoir **Docker Desktop** installé.

1. **Cloner le projet**
   ```bash
   git clone https://github.com/leo-rb/urban-pulse-mlops.git
   cd urban-pulse-mlops
   docker-compose up -d --build
   ``` 

2. Entrainer le modèle (une fois les conteneurs lancés)
```bash
   docker exec urban_generator python src/modeling/train_model.py
```

3. Accéder aux interfaces
Dashboard : http://localhost:8501

MLflow UI : http://localhost:5000

MinIO Console : http://localhost:9001

API Docs : http://localhost:8000/docs


🛠️ Stack Technologique

Langage : Python 3.10+

Containerization : Docker & Docker Compose

ML Framework : Scikit-Learn, Pandas

MLOps : MLflow, MinIO (S3)

Backend/Frontend : FastAPI, Streamlit, PostgreSQL


### 🧹 Comment arrêter proprement ?

Quand tu as fini de jouer :

1.  **Tout éteindre :**
    ```powershell
    docker-compose down
    ```
    *(Tes données seront conservées grâce aux volumes Docker).*

2.  **Si tu veux tout raser (Attention, perte de données) :**
    ```powershell
    docker-compose down -v
    ```