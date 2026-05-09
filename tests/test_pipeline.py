import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ingest_rankings import (
    build_catalog_metadata,
    build_payload,
    enrich_candidates,
    compute_rankings,
    extract_openrouter_model,
    load_json,
    write_json,
)


class PipelineTests(unittest.TestCase):
    def test_extract_openrouter_model_normalizes_costs_context_and_source(self):
        openrouter = {
            "data": [
                {
                    "id": "provider/model-a",
                    "name": "Provider Model A",
                    "context_length": 123456,
                    "pricing": {"prompt": "0.00000025", "completion": "0.0000015"},
                    "links": {"details": "/api/v1/models/provider/model-a/endpoints"},
                }
            ]
        }

        result = extract_openrouter_model(openrouter, "provider/model-a")

        self.assertEqual(result["openrouter_id"], "provider/model-a")
        self.assertEqual(result["context_length"], 123456)
        self.assertAlmostEqual(result["input_per_million"], 0.25)
        self.assertAlmostEqual(result["output_per_million"], 1.5)
        self.assertEqual(result["blended_per_million"], 1.75)
        self.assertIsNone(result["created_at"])
        self.assertIn("openrouter.ai/provider/model-a", result["sources"][0]["url"])

    def test_enrich_candidates_prefers_live_openrouter_prices_over_static_values(self):
        candidates = [
            {
                "id": "model-a",
                "name": "Model A",
                "provider": "Provider",
                "provider_group": "Open",
                "openrouter_id": "provider/model-a",
                "parameter_note": "120B",
                "intelligence_seed": 80,
                "official_sources": [],
            }
        ]
        openrouter = {
            "data": [
                {
                    "id": "provider/model-a",
                    "name": "Provider Model A",
                    "context_length": 1000,
                    "pricing": {"prompt": "0.00000010", "completion": "0.00000040"},
                    "links": {"details": "/api/v1/models/provider/model-a/endpoints"},
                }
            ]
        }

        enriched = enrich_candidates(candidates, openrouter, benchmark_signals={})

        self.assertEqual(enriched[0]["context"], "1K")
        self.assertEqual(enriched[0]["input_per_million"], 0.1)
        self.assertEqual(enriched[0]["output_per_million"], 0.4)
        self.assertEqual(enriched[0]["blended_per_million"], 0.5)
        self.assertEqual(enriched[0]["intelligence"], 80)

    def test_compute_rankings_creates_three_sorted_lists_and_value_index(self):
        models = [
            {"id": "smart-expensive", "intelligence": 98, "blended_per_million": 100},
            {"id": "smart-cheap", "intelligence": 90, "blended_per_million": 2},
            {"id": "cheap", "intelligence": 70, "blended_per_million": 0.1},
        ]

        rankings = compute_rankings(models)

        self.assertEqual(rankings["intelligence"][0], "smart-expensive")
        self.assertEqual(rankings["value"][0], "smart-cheap")
        self.assertEqual(rankings["cheap"][0], "cheap")
        self.assertEqual(len(rankings["value_index"]), 3)
        self.assertEqual(rankings["value_index"]["smart-cheap"], 100)
        self.assertEqual(rankings["value_index"]["cheap"], 0)

    def test_missing_openrouter_lookup_serializes_as_null_cost(self):
        candidates = [
            {
                "id": "model-a",
                "name": "Model A",
                "provider": "Provider",
                "provider_group": "Open",
                "openrouter_id": "provider/missing",
                "parameter_note": "120B",
                "intelligence_seed": 80,
                "official_sources": [],
            }
        ]

        enriched = enrich_candidates(candidates, {"data": []}, benchmark_signals={})

        self.assertIsNone(enriched[0]["blended_per_million"])
        self.assertEqual(compute_rankings(enriched)["cheap"], [])

    def test_offline_build_uses_committed_rankings_without_network_when_cache_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            candidates_path = data_dir / "candidates.json"
            candidates_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "model-a",
                            "name": "Model A",
                            "provider": "Provider",
                            "provider_group": "Open",
                            "openrouter_id": "provider/model-a",
                            "parameter_note": "120B",
                            "intelligence_seed": 84,
                            "official_sources": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (data_dir / "rankings.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-05-09T00:00:00+00:00",
                        "models": [
                            {
                                "id": "model-a",
                                "name": "Model A",
                                "openrouter_id": "provider/model-a",
                                "context_length": 1000,
                                "input_per_million": 0.1,
                                "output_per_million": 0.4,
                                "blended_per_million": 0.5,
                                "knowledge_cutoff": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("scripts.ingest_rankings.DATA_DIR", data_dir), patch(
                "scripts.ingest_rankings.fetch_json", side_effect=AssertionError("network used")
            ):
                payload = build_payload(candidates_path, offline=True)

        self.assertEqual(payload["source_status"][0]["status"], "from-generated")
        self.assertEqual(payload["models"][0]["blended_per_million"], 0.5)

    def test_write_json_rejects_non_standard_json_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_json(Path(d) / "bad.json", {"cost": math.inf})

    def test_catalog_metadata_flags_unranked_frontier_candidates(self):
        payload = {
            "data": [
                {
                    "id": "provider/new-frontier",
                    "name": "Provider New Frontier 120B",
                    "description": "A 120B reasoning model for agentic work.",
                    "created": 1778247440,
                    "context_length": 262144,
                    "pricing": {"prompt": "0.0000001", "completion": "0.0000005"},
                }
            ]
        }

        catalog = build_catalog_metadata(payload, candidates=[])

        self.assertEqual(catalog["discovery_alerts"][0]["id"], "provider/new-frontier")
        self.assertEqual(catalog["discovery_alerts"][0]["blended_per_million"], 0.6)

    def test_load_json_reads_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(load_json(path), {"ok": True})


if __name__ == "__main__":
    unittest.main()
