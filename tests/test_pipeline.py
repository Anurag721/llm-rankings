import json
import tempfile
import unittest
from pathlib import Path

from scripts.ingest_rankings import (
    enrich_candidates,
    compute_rankings,
    extract_openrouter_model,
    load_json,
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
        self.assertEqual(rankings["value_index"]["cheap"], 100)

    def test_load_json_reads_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(load_json(path), {"ok": True})


if __name__ == "__main__":
    unittest.main()
