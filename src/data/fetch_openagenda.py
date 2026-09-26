"""
Récupération des événements culturels depuis l'API Open Agenda (OpenDataSoft).

Fonctionnalités :
- Pagination automatique (au-delà de la limite de résultats par page de l'API).
- Retries avec backoff exponentiel en cas d'erreur réseau ou HTTP 5xx/429.
- Encodage correct des paramètres de requête (guillemets, accents, espaces...).
- Nettoyage des données (valeurs manquantes, doublons, champs vides).
- Enrichissement (normalisation des villes, parsing des coordonnées, indicateur
  de complétude de la localisation, fraîcheur des événements).
"""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from urllib3.util.retry import Retry

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_RAW_DIR = Path("data/raw")
API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "evenements-publics-openagenda/records"
)
PAGE_SIZE = 100
MAX_HISTORY_DAYS = 365
REQUIRED_FIELDS = ["title", "description"]


class OpenAgendaAPIError(Exception):
    """Levée lorsque l'API Open Agenda est inaccessible après plusieurs tentatives."""


def _build_session() -> requests.Session:
    """Session HTTP avec retries automatiques sur les erreurs transitoires."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _build_where_clause(city: str, since_days: int) -> str:
    """
    Construit la clause de filtrage ODSQL, avec échappement des guillemets
    simples pour éviter toute erreur de requête (ex. villes composées).
    L'encodage URL des paramètres est ensuite délégué à `requests`.
    """
    since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%Y-%m-%d")
    safe_city = city.replace("'", "\\'")
    return f"location_city='{safe_city}' AND lastdate_end >= date'{since_date}'"


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
)
def _fetch_page(session: requests.Session, params: dict[str, Any]) -> dict:
    """Récupère une page de résultats, avec retry/backoff en cas d'erreur réseau."""
    response = session.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_records(
    city: str = "Bordeaux",
    since_days: int = MAX_HISTORY_DAYS,
    max_records: int = 500,
    page_size: int = PAGE_SIZE,
) -> list[dict]:
    """
    Récupère tous les événements correspondant aux filtres, en paginant
    automatiquement (l'API OpenDataSoft limite chaque page à `page_size`).
    """
    session = _build_session()
    where_clause = _build_where_clause(city, since_days)

    all_records: list[dict] = []
    offset = 0

    while len(all_records) < max_records:
        limit = min(page_size, max_records - len(all_records))
        params = {
            "limit": limit,
            "offset": offset,
            "where": where_clause,
            "order_by": "lastdate_begin desc",
        }
        try:
            data = _fetch_page(session, params)
        except requests.exceptions.RequestException as exc:
            raise OpenAgendaAPIError(
                f"Échec de récupération après plusieurs tentatives (offset={offset}) : {exc}"
            ) from exc

        records = data.get("results", [])
        if not records:
            break

        all_records.extend(records)
        offset += len(records)

        if len(records) < limit:
            break

    # Garde-fou : certaines API peuvent ignorer le paramètre `limit` demandé,
    # on tronque donc explicitement au nombre maximal souhaité.
    all_records = all_records[:max_records]

    logger.info("Nombre total d'événements récupérés : %d", len(all_records))
    return all_records


def parse_records(records: list[dict]) -> pd.DataFrame:
    """Transforme les enregistrements bruts de l'API en DataFrame structuré."""
    cleaned_records = []
    for item in records:
        title = item.get("title_fr") or item.get("title", "")
        description = (
            item.get("description_fr")
            or item.get("longdescription_fr")
            or item.get("description", "")
        )
        cleaned_records.append(
            {
                "uid": item.get("uid"),
                "title": title,
                "description": description,
                "date_start": item.get("firstdate_begin"),
                "date_end": item.get("lastdate_end"),
                "location_name": item.get("location_name"),
                "location_address": item.get("location_address"),
                "location_city": item.get("location_city"),
                "location_postalcode": item.get("location_postalcode"),
                "location_coordinates": json.dumps(item.get("location_coordinates", {})),
            }
        )
    return pd.DataFrame(cleaned_records)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le jeu de données :
    - suppression des lignes sans titre/description (champs obligatoires),
    - suppression des doublons (par uid),
    - normalisation des espaces superflus.
    """
    if df.empty:
        return df

    df = df.copy()

    for field in REQUIRED_FIELDS:
        df[field] = df[field].fillna("").astype(str).str.strip()

    for field in REQUIRED_FIELDS:
        df = df[df[field] != ""]

    if "uid" in df.columns:
        df = df.drop_duplicates(subset=["uid"], keep="first")

    return df.reset_index(drop=True)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit et valide le jeu de données :
    - normalisation de la casse des noms de ville,
    - extraction latitude/longitude depuis les coordonnées JSON,
    - marquage des lignes avec des données de localisation incomplètes,
    - parsing de la date de fin pour analyse de fraîcheur.
    """
    if df.empty:
        return df

    df = df.copy()

    df["location_city"] = (
        df["location_city"].fillna("Non renseigné").astype(str).str.strip().str.title()
    )

    def _extract_coord(raw: Optional[str], key: str) -> Optional[float]:
        try:
            parsed = json.loads(raw) if raw else {}
            return parsed.get(key)
        except (json.JSONDecodeError, TypeError):
            return None

    df["latitude"] = df["location_coordinates"].apply(lambda v: _extract_coord(v, "lat"))
    df["longitude"] = df["location_coordinates"].apply(lambda v: _extract_coord(v, "lon"))

    df["has_complete_location"] = (
        df["location_name"].notna() & df["location_address"].notna() & df["latitude"].notna()
    )

    df["date_end_parsed"] = pd.to_datetime(df["date_end"], errors="coerce", utc=True)

    return df


def fetch_events(
    city: str = "Bordeaux", since_days: int = MAX_HISTORY_DAYS, max_records: int = 500
) -> pd.DataFrame:
    """Pipeline complet : récupération, parsing, nettoyage et enrichissement."""
    records = fetch_all_records(city=city, since_days=since_days, max_records=max_records)

    if not records:
        logger.warning("Aucun événement trouvé avec ces filtres.")
        return pd.DataFrame()

    df = parse_records(records)
    df = clean_dataframe(df)
    df = enrich_dataframe(df)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW_DIR / "events_raw.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Extraction réussie : %d événements sauvegardés dans %s", len(df), output_path)
    return df


if __name__ == "__main__":
    fetch_events(city="Bordeaux")
