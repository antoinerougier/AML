# Utilise une image Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /AML

# Copier les fichiers du projet
COPY . .

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Spécifie la commande à exécuter
CMD ["python", "main.py"]
