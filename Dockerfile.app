FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système pour psycopg2
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Installation des librairies Python
RUN pip install pandas sqlalchemy psycopg2-binary python-dotenv mlflow==2.9.2 boto3
# Copie du code
COPY src/ /app/src/

CMD ["python", "src/ingestion/mock_generator.py"]