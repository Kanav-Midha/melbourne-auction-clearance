"""API contract tests.

These are the tests that catch training/serving skew. The model can be perfect
offline and still be wrong at request time if the online feature row disagrees
with the offline one about a column order, a dtype, or a category level.
"""
from __future__ import annotations

import pytest

from tests.conftest import requires_model

VALID_PAYLOAD = {
    "suburb": "Northcote",
    "auction_date": "2025-03-15",
    "property_type": "h",
    "bedrooms": 3,
    "bathrooms": 1,
    "car_spaces": 1,
    "land_size_sqm": 372.0,
    "building_area_sqm": 128.0,
    "year_built": 1925,
    "guide_price_low": 1480000,
    "guide_price_high": 1620000,
    "agency": "Nelson Alexander",
}

pytestmark = requires_model


class TestHealth:
    def test_health_reports_ready(self, api_client):
        body = api_client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] and body["context_loaded"]
        assert body["n_suburbs"] == 100

    def test_model_info_exposes_held_out_metrics(self, api_client):
        body = api_client.get("/model/info").json()
        assert body["n_features"] == 31
        assert 0.5 < body["test_metrics"]["roc_auc"] < 0.9
        assert len(body["top_features"]) == 10


class TestPredict:
    def test_returns_a_calibrated_probability(self, api_client):
        r = api_client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["clearance_probability"] <= 1.0
        assert body["predicted_outcome"] in {"likely_sell", "uncertain", "likely_pass_in"}

    def test_explains_the_drivers(self, api_client):
        body = api_client.post("/predict", json=VALID_PAYLOAD).json()
        drivers = body["drivers"]
        assert drivers["expected_value"] > 0
        assert drivers["suburb_clearance_l4w"] is not None
        assert drivers["rba_cash_rate"] > 0

    def test_is_deterministic(self, api_client):
        a = api_client.post("/predict", json=VALID_PAYLOAD).json()
        b = api_client.post("/predict", json=VALID_PAYLOAD).json()
        assert a["clearance_probability"] == b["clearance_probability"]

    def test_overpricing_lowers_the_probability(self, api_client):
        """The core business signal: guide the same house 60% higher."""
        cheap = api_client.post("/predict", json=VALID_PAYLOAD).json()
        dear = api_client.post("/predict", json={
            **VALID_PAYLOAD,
            "guide_price_low": 2_400_000,
            "guide_price_high": 2_600_000,
        }).json()
        assert dear["clearance_probability"] < cheap["clearance_probability"]
        assert dear["drivers"]["guide_vs_expected_value_pct"] > \
               cheap["drivers"]["guide_vs_expected_value_pct"]

    def test_suburb_is_case_insensitive(self, api_client):
        a = api_client.post("/predict", json=VALID_PAYLOAD).json()
        b = api_client.post("/predict", json={**VALID_PAYLOAD, "suburb": "  northcote "}).json()
        assert a["clearance_probability"] == b["clearance_probability"]

    def test_optional_fields_may_be_omitted(self, api_client):
        payload = {k: v for k, v in VALID_PAYLOAD.items()
                   if k not in {"land_size_sqm", "building_area_sqm", "year_built", "agency"}}
        assert api_client.post("/predict", json=payload).status_code == 200


class TestValidation:
    def test_unknown_suburb_is_rejected_not_guessed(self, api_client):
        r = api_client.post("/predict", json={**VALID_PAYLOAD, "suburb": "Bondi"})
        assert r.status_code == 422
        assert "unknown suburb" in r.json()["detail"].lower()

    def test_inverted_price_guide_is_rejected(self, api_client):
        r = api_client.post("/predict", json={
            **VALID_PAYLOAD, "guide_price_low": 2_000_000, "guide_price_high": 1_000_000})
        assert r.status_code == 422

    def test_absurd_bedroom_count_is_rejected(self, api_client):
        assert api_client.post(
            "/predict", json={**VALID_PAYLOAD, "bedrooms": 99}).status_code == 422

    def test_bad_property_type_is_rejected(self, api_client):
        assert api_client.post(
            "/predict", json={**VALID_PAYLOAD, "property_type": "castle"}).status_code == 422

    def test_missing_required_field_is_rejected(self, api_client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "suburb"}
        assert api_client.post("/predict", json=payload).status_code == 422

    def test_stale_context_produces_a_warning(self, api_client):
        r = api_client.post("/predict", json={**VALID_PAYLOAD, "auction_date": "2027-06-12"})
        assert r.status_code == 200
        assert any("stale" in w for w in r.json()["warnings"])


class TestBatch:
    def test_scores_a_campaign_and_summarises(self, api_client):
        r = api_client.post("/predict/batch", json={
            "properties": [
                VALID_PAYLOAD,
                {**VALID_PAYLOAD, "suburb": "Werribee", "guide_price_low": 600000,
                 "guide_price_high": 650000},
                {**VALID_PAYLOAD, "suburb": "Toorak", "guide_price_low": 3800000,
                 "guide_price_high": 4100000},
            ]
        })
        assert r.status_code == 200
        body = r.json()
        assert len(body["predictions"]) == 3
        s = body["summary"]
        assert s["n"] == 3
        assert 0 <= s["expected_sales"] <= 3
        assert s["likely_sell"] + s["uncertain"] + s["likely_pass_in"] == 3

    def test_batch_matches_single_predictions(self, api_client):
        single = api_client.post("/predict", json=VALID_PAYLOAD).json()
        batch = api_client.post(
            "/predict/batch", json={"properties": [VALID_PAYLOAD]}).json()
        assert batch["predictions"][0]["clearance_probability"] == \
               single["clearance_probability"]

    def test_empty_batch_is_rejected(self, api_client):
        assert api_client.post(
            "/predict/batch", json={"properties": []}).status_code == 422


class TestReference:
    def test_suburbs_endpoint_lists_coverage(self, api_client):
        body = api_client.get("/suburbs").json()
        assert body["count"] == 100
        assert "Northcote" in body["suburbs"]
