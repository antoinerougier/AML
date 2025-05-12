import os
import requests
import zipfile
import io
from pathlib import Path

# URL du fichier ZIP
URL_ZIP = "https://minio.lab.sspcloud.fr/arougier/AML/archive.zip"
zip_path = os.environ.get("zip_path", URL_ZIP)

# Requête pour télécharger le fichier
response = requests.get(zip_path)
response.raise_for_status()

# Aller à la racine du projet (ex: AML/) et créer le dossier 'data' là
project_root = Path(__file__).resolve().parents[2]  # <-- remonte deux niveaux pour sortir de src/
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Extraire le contenu du zip dans AML/data/
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(data_dir)

print(f"Données extraites dans : {data_dir}")
