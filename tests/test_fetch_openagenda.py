"""
Tests unitaires du module de récupération des événements Open Agenda.

Couvre : construction des paramètres/URL, pagination, gestion des erreurs
réseau, nettoyage et enrichissement des données.
"""
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from src.data.fetch_openagenda import (
    OpenAgendaAPIError,
    _build_where_clause,
    clean_dataframe,
    enrich_dataframe,
    fetch_all_records,
    parse_records,
)

SAMPLE_RECORD = {
    "uid": "1",
    "title_fr": "Concert de Jazz",
    "description_fr": "Un super concert de jazz en plein air.",
    "firstdate_begin": "2026-10-01T20:00:00+00:00",
    "lastdate_end": "2026-10-01T23:00:00+00:00",
    "location_name": "Salle Jazz",
    "location_address": "1 rue de la Musique",
    "location_city": "bordeaux",
    "location_postalcode": "33000",
    "location_coordinates": {"lat": 44.84, "lon": -0.58},
}


# --- Construction de la requête (encodage des paramètres) ---------------------


def test_build_where_clause_escapes_quotes():
    clause = _build_where_clause("L'Isle-d'Abeau", since_days=365)
    assert "\\'" in clause
    assert "location_city=" in clause
    assert "lastdate_end >=" in clause


def test_build_where_clause_includes_since_date():
    clause = _build_where_clause("Bordeaux", since_days=30)
    assert "date'" in clause


# --- Parsing ------------------------------------------------------------------


def test_parse_records_extracts_expected_fields():
    df = parse_records([SAMPLE_RECORD])
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Concert de Jazz"
    assert df.iloc[0]["location_city"] == "bordeaux"


def test_parse_records_handles_missing_optional_fields():
    minimal_record = {"uid": "2", "title": "Titre simple", "description": "Description simple"}
    df = parse_records([minimal_record])
    assert len(df) == 1
    assert df.iloc[0]["location_city"] is None


# --- Nettoyage ------------------------------------------------------------------


def test_clean_dataframe_drops_missing_required_fields():
    df = pd.DataFrame(
        [
            {"uid": "1", "title": "Titre valide", "description": "Description valide"},
            {"uid": "2", "title": "", "description": "Sans titre"},
            {"uid": "3", "title": "Sans description", "description": ""},
        ]
    )
    cleaned = clean_dataframe(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["uid"] == "1"


def test_clean_dataframe_drops_duplicates():
    df = pd.DataFrame(
        [
            {"uid": "1", "title": "A", "description": "B"},
            {"uid": "1", "title": "A", "description": "B"},
        ]
    )
    cleaned = clean_dataframe(df)
    assert len(cleaned) == 1


def test_clean_dataframe_handles_empty_input():
    assert clean_dataframe(pd.DataFrame()).empty


# --- Enrichissement --------------------------------------------------------------


def test_enrich_dataframe_extracts_coordinates_and_flags_completeness():
    df = parse_records([SAMPLE_RECORD])
    enriched = enrich_dataframe(df)
    assert enriched.iloc[0]["latitude"] == 44.84
    assert enriched.iloc[0]["longitude"] == -0.58
    assert bool(enriched.iloc[0]["has_complete_location"]) is True
    assert enriched.iloc[0]["location_city"] == "Bordeaux"  # normalisation title-case


def test_enrich_dataframe_handles_missing_coordinates_gracefully():
    df = pd.DataFrame(
        [
            {
                "uid": "2",
                "title": "Sans coordonnées",
                "description": "Test",
                "location_name": None,
                "location_address": None,
                "location_city": None,
                "location_coordinates": "",
                "date_end": None,
            }
        ]
    )
    enriched = enrich_dataframe(df)
    assert enriched.iloc[0]["latitude"] is None
    assert bool(enriched.iloc[0]["has_complete_location"]) is False
    assert enriched.iloc[0]["location_city"] == "Non Renseigné"


# --- Pagination & robustesse réseau -----------------------------------------------


@patch("src.data.fetch_openagenda._fetch_page")
def test_fetch_all_records_paginates_until_short_page(mock_fetch_page):
    mock_fetch_page.side_effect = [
        {"results": [SAMPLE_RECORD] * 100},
        {"results": [SAMPLE_RECORD] * 40},
    ]
    records = fetch_all_records(city="Bordeaux", max_records=1000, page_size=100)
    assert len(records) == 140
    assert mock_fetch_page.call_count == 2  # la page de 40 < 100 arrête la pagination


@patch("src.data.fetch_openagenda._fetch_page")
def test_fetch_all_records_respects_max_records(mock_fetch_page):
    mock_fetch_page.return_value = {"results": [SAMPLE_RECORD] * 100}
    records = fetch_all_records(city="Bordeaux", max_records=150, page_size=100)
    assert len(records) == 150


@patch("src.data.fetch_openagenda._fetch_page")
def test_fetch_all_records_stops_on_empty_page(mock_fetch_page):
    mock_fetch_page.return_value = {"results": []}
    records = fetch_all_records(city="Bordeaux", max_records=500)
    assert records == []
    assert mock_fetch_page.call_count == 1


@patch("src.data.fetch_openagenda._fetch_page")
def test_fetch_all_records_raises_explicit_error_on_network_failure(mock_fetch_page):
    mock_fetch_page.side_effect = requests.exceptions.ConnectionError("boom")

    with pytest.raises(OpenAgendaAPIError):
        fetch_all_records(city="Bordeaux")
