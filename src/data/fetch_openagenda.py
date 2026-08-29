import os
import json
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_RAW_DIR = Path("data/raw")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Endpoint public OpenData v2 pour les événements en Nouvelle-Aquitaine / Bordeaux
# Si tu possèdes un slug d'agenda spécifique, remplace l'URL ci-dessous.
API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

def fetch_events(limit: int = 100, city: str = "Bordeaux") -> pd.DataFrame:
    """Récupère les événements récents depuis l'API Open Agenda."""
    params = {
        "limit": limit,
        "where": f"location_city='{city}'",
        "order_by": "lastdate_begin desc"
    }
    
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    records = data.get("results", [])
    
    if not records:
        print("Aucun événement trouvé avec ces filtres.")
        return pd.DataFrame()
    
    # Extraction et standardisation des champs utiles
    cleaned_records = []
    for item in records:
        title = item.get("title_fr") or item.get("title", "")
        description = item.get("description_fr") or item.get("longdescription_fr") or item.get("description", "")
        
        cleaned_records.append({
            "uid": item.get("uid"),
            "title": title,
            "description": description,
            "date_start": item.get("firstdate_begin"),
            "date_end": item.get("lastdate_end"),
            "location_name": item.get("location_name"),
            "location_address": item.get("location_address"),
            "location_city": item.get("location_city"),
            "location_postalcode": item.get("location_postalcode"),
            "location_coordinates": json.dumps(item.get("location_coordinates", {}))
        })
        
    df = pd.DataFrame(cleaned_records)
    
    # Nettoyage des descriptions vides
    df = df[df["description"].str.strip() != ""]
    df = df[df["title"].str.strip() != ""]
    
    output_path = DATA_RAW_DIR / "events_raw.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Extraction réussie : {len(df)} événements sauvegardés dans {output_path}")
    return df

if __name__ == "__main__":
    fetch_events(limit=100, city="Bordeaux")