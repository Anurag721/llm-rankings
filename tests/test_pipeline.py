import re
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.ingest_rankings import (
    build_catalog_metadata,
    build_payload,
    compute_rankings,
    dollars_per_million,
    enrich_candidates,
    extract_openrouter_model,
    format_context,
    is_frontier_discovery,
    load_json,
    now_iso,
    summarize_benchmark_page,
    unix_to_iso,
    write_json,
)


class TestExtractOpenRouterModel(unittest.TestCase):
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

    def test_extract_openrouter_model_returns_none_for_missing_id(self):
        openrouter = {"data": [{"id": "other/model", "name": "Other"}]}
        result = extract_openrouter_model(openrouter, "provider/missing")
        self.assertIsNone(result)

    def test_extract_openrouter_model_handles_missing_pricing(self):
        openrouter = {
            "data": [
                {
                    "id": "provider/free",
                    "name": "Free Model",
                    "context_length": 1000,
                    "pricing": {"prompt": "0", "completion": "0"},
                }
            ]
        }
        result = extract_openrouter_model(openrouter, "provider/free")
        self.assertEqual(result["input_per_million"], 0.0)
        self.assertEqual(result["output_per_million"], 0.0)
        self.assertEqual(result["blended_per_million"], 0.0)

    def test_extract_openrouter_model_handles_null_pricing(self):
        openrouter = {
            "data": [
                {
                    "id": "provider/null-price",
                    "name": "Null Price Model",
                    "context_length": 1000,
                    "pricing": {"prompt": None, "completion": None},
                }
            ]
        }
        result = extract_openrouter_model(openrouter, "provider/null-price")
        self.assertIsNone(result["input_per_million"])
        self.assertIsNone(result["output_per_million"])
        self.assertIsNone(result["blended_per_million"])


class TestEnrichCandidates(unittest.TestCase):
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

    def test_enrich_candidates_merges_official_and_live_sources(self):
        candidates = [
            {
                "id": "model-b",
                "name": "Model B",
                "provider": "Provider",
                "provider_group": "Open",
                "openrouter_id": "provider/model-b",
                "parameter_note": "10B",
                "intelligence_seed": 70,
                "official_sources": [
                    {"label": "Official Blog", "url": "https://example.com/blog"}
                ],
            }
        ]
        openrouter = {
            "data": [
                {
                    "id": "provider/model-b",
                    "name": "Model B",
                    "context_length": 5000,
                    "pricing": {"prompt": "0.00000010", "completion": "0.00000010"},
                }
            ]
        }
        enriched = enrich_candidates(candidates, openrouter, benchmark_signals={})
        sources = enriched[0]["sources"]
        # OpenRouter source comes first, then official sources
        self.assertEqual(sources[0]["label"], "OpenRouter")
        self.assertEqual(sources[1]["label"], "Official Blog")

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
        self.assertEqual(enriched[0]["context"], "Unknown")
        self.assertEqual(enriched[0]["sources"][0]["type"], "pricing-warning")


class TestComputeRankings(unittest.TestCase):
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

    def test_compute_rankings_excludes_low_intelligence_from_value(self):
        models = [
            {"id": "high-intel", "intelligence": 95, "blended_per_million": 50},
            {"id": "low-intel", "intelligence": 70, "blended_per_million": 1},
        ]
        rankings = compute_rankings(models, intelligence_cutoff=83)
        self.assertIn("high-intel", rankings["value"])
        self.assertNotIn("low-intel", rankings["value"])

    def test_compute_rankings_value_index_scaled_to_100(self):
        models = [
            {"id": "a", "intelligence": 90, "blended_per_million": 1},
            {"id": "b", "intelligence": 90, "blended_per_million": 10},
        ]
        rankings = compute_rankings(models)
        self.assertEqual(rankings["value_index"]["a"], 100)
        self.assertLess(rankings["value_index"]["b"], 100)

    def test_compute_rankings_with_null_costs(self):
        models = [
            {"id": "no-cost", "intelligence": 90, "blended_per_million": None},
            {"id": "has-cost", "intelligence": 85, "blended_per_million": 5.0},
        ]
        rankings = compute_rankings(models)
        self.assertIn("has-cost", rankings["cheap"])
        self.assertNotIn("no-cost", rankings["cheap"])


class TestIntelligenceScore(unittest.TestCase):
    def test_score_uses_seed_when_no_benchmark_signal(self):
        candidate = {"intelligence_seed": 85}
        score = enrich_candidates.__wrapped__._intelligence_score if hasattr(enrich_candidates, '__wrapped__') else None
        # Test via enrich_candidates behavior
        result = enrich_candidates(
            [{"id": "m", "name": "M", "provider": "P", "provider_group": "G",
              "openrouter_id": "m", "parameter_note": "10B", "intelligence_seed": 85}],
            {"data": []},
            benchmark_signals={},
        )
        self.assertEqual(result[0]["intelligence"], 85)

    def test_score_blends_seed_and_signal(self):
        # seed=80, signal=100, blend = 0.55*80 + 0.45*100 = 44 + 45 = 89
        candidates = [
            {"id": "m", "name": "M", "provider": "P", "provider_group": "G",
             "openrouter_id": "m", "parameter_note": "10B", "intelligence_seed": 80}
        ]
        result = enrich_candidates(candidates, {"data": []}, benchmark_signals={"m": 100})
        self.assertEqual(result[0]["intelligence"], 89)


class TestFormatContext(unittest.TestCase):
    def test_format_context_various_sizes(self):
        self.assertEqual(format_context(None), "Unknown")
        self.assertEqual(format_context(0), "Unknown")
        self.assertEqual(format_context(500), "0K")
        self.assertEqual(format_context(1000), "1K")
        self.assertEqual(format_context(1_500_000), "1.5M")
        self.assertEqual(format_context(1_048_576), "1.04858M")


class TestDollarsPerMillion(unittest.TestCase):
    def test_conversion(self):
        self.assertEqual(dollars_per_million("0.00000025"), 0.25)
        self.assertEqual(dollars_per_million("0.0000015"), 1.5)
        self.assertIsNone(dollars_per_million(None))
        self.assertIsNone(dollars_per_million(""))
        self.assertEqual(dollars_per_million(0.000001), 1.0)


class TestUnixToIso(unittest.TestCase):
    def test_valid_timestamp(self):
        result = unix_to_iso(1778247440)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("+00:00"))

    def test_invalid_values(self):
        self.assertIsNone(unix_to_iso(None))
        self.assertIsNone(unix_to_iso(""))
        self.assertIsNone(unix_to_iso("not-a-number"))


class TestSummarizeBenchmarkPage(unittest.TestCase):
    def test_empty_html(self):
        self.assertEqual(
            summarize_benchmark_page(""),
            "No page content available; using curated benchmark seed scores."
        )

    def test_html_with_title(self):
        html = "<html><head><title>  LLM Leaderboard  </title></head></html>"
        result = summarize_benchmark_page(html)
        self.assertIn("LLM Leaderboard", result)

    def test_html_without_title(self):
        html = "<html><body>Hello</body></html>"
        result = summarize_benchmark_page(html)
        self.assertIn("page fetched", result)


class TestIsFrontierDiscovery(unittest.TestCase):
    def setUp(self):
        import re as _re
        self.param_re = _re.compile(
            r"\b(?:[1-9]\d{2,}(?:\.\d+)?\s*B|[1-9](?:\.\d+)?\s*T)\b", _re.I
        )
        self.signal_re = _re.compile(
            r"\b(frontier|flagship|reasoning|agentic|coding agent|foundation model|large language model)\b",
            _re.I,
        )
        self.providers = {"anthropic", "google", "openai", "x-ai"}

    def test_detects_by_parameter_size(self):
        model = {"id": "prov/120B-model", "name": "120B Model", "description": ""}
        self.assertTrue(is_frontier_discovery(
            model, param_re=self.param_re, signal_re=self.signal_re,
            frontier_providers=self.providers,
        ))

    def test_detects_by_signal_word_with_known_provider(self):
        model = {
            "id": "openai/new-model",
            "name": "OpenAI Reasoning Model",
            "description": "A frontier reasoning model",
        }
        self.assertTrue(is_frontier_discovery(
            model, param_re=self.param_re, signal_re=self.signal_re,
            frontier_providers=self.providers,
        ))

    def test_unknown_provider_no_signal(self):
        model = {
            "id": "unknown/small-model",
            "name": "Small Model",
            "description": "A small model",
        }
        self.assertFalse(is_frontier_discovery(
            model, param_re=self.param_re, signal_re=self.signal_re,
            frontier_providers=self.providers,
        ))

    def test_known_provider_signal_without_size(self):
        model = {
            "id": "google/tiny-but-smart",
            "name": "Tiny",
            "description": "A foundation model for research",
        }
        self.assertTrue(is_frontier_discovery(
            model, param_re=self.param_re, signal_re=self.signal_re,
            frontier_providers=self.providers,
        ))


class TestBuildCatalogMetadata(unittest.TestCase):
    def setUp(self):
        self.param_re = re.compile(
            r"\b(?:[1-9]\d{2,}(?:\.\d+)?\s*B|[1-9](?:\.\d+)?\s*T)\b", re.I
        )
        self.signal_re = re.compile(
            r"\b(frontier|flagship|reasoning|agentic|coding agent|foundation model|large language model)\b",
            re.I,
        )
        self.frontier_providers = {"anthropic", "google", "openai", "x-ai"}

    def test_flags_unranked_frontier_candidates(self):
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

        catalog = build_catalog_metadata(
            payload,
            candidates=[],
            param_re=self.param_re,
            signal_re=self.signal_re,
            frontier_providers=self.frontier_providers,
        )

        self.assertEqual(len(catalog["discovery_alerts"]), 1)
        self.assertEqual(catalog["discovery_alerts"][0]["id"], "provider/new-frontier")
        self.assertEqual(catalog["discovery_alerts"][0]["blended_per_million"], 0.6)

    def test_empty_candidates_still_counts_models(self):
        payload = {"data": [{"id": "m1", "name": "M1", "created": 1000}]}
        catalog = build_catalog_metadata(
            payload,
            candidates=[],
            param_re=self.param_re,
            signal_re=self.signal_re,
            frontier_providers=self.frontier_providers,
        )
        self.assertEqual(catalog["openrouter_model_count"], 1)
        self.assertEqual(catalog["ranked_candidate_count"], 0)


class TestLoadJsonWriteJson(unittest.TestCase):
    def test_write_and_load_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.json"
            data = {"key": "value", "number": 42}
            write_json(path, data)
            loaded = load_json(path)
            self.assertEqual(loaded, data)

    def test_write_json_rejects_inf(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_json(Path(d) / "bad.json", {"cost": math.inf})

    def test_load_json_reads_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(load_json(path), {"ok": True})


class TestBuildPayloadOffline(unittest.TestCase):
    def test_offline_build_uses_committed_rankings_without_network(self):
        import yaml as _yaml

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
            cfg_path = data_dir / "config.yaml"
            cfg = {
                "pipeline": {
                    "openrouter_url": "https://openrouter.ai/api/v1/models",
                    "user_agent": "LLMSignalBot/1.0",
                    "stale_after_hours": 24,
                    "review_after_hours": 168,
                    "discovery_limit": 12,
                    "intelligence_cutoff_for_value": 83,
                    "max_ranked_results": 6,
                    "value_blend_ratio": 0.55,
                    "benchmark_signal_ratio": 0.45,
                    "min_cost_for_value": 0.05,
                    "fetch_timeout_seconds": 30,
                    "text_fetch_timeout_seconds": 20,
                    "max_retries": 3,
                    "retry_backoff_seconds": 2,
                },
                "paths": {
                    "candidates": str(candidates_path),
                    "output": str(data_dir / "rankings.json"),
                    "cache_dir": str(data_dir / ".cache"),
                },
                "benchmark_sources": [],
            }
            cfg_path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")

            from scripts.ingest_rankings import build_payload

            payload = build_payload(candidates_path, cfg=cfg, offline=True)

        assert payload["source_status"][0]["status"] == "from-generated"
        assert payload["models"][0]["blended_per_million"] == 0.5


class TestValidatePayload(unittest.TestCase):
    """Tests for the improved validate_rankings.py."""

    def _make_payload(self, overrides=None):
        payload = {
            "generated_at": "2026-05-09T14:50:58+00:00",
            "scope": "test scope",
            "methodology": {"intelligence": "seed-based", "cost": "orp", "value": "intel/cost"},
            "source_status": [
                {
                    "id": "openrouter-models",
                    "label": "OpenRouter",
                    "url": "https://openrouter.ai/api/v1/models",
                    "type": "pricing",
                    "status": "fetched",
                    "fetched_at": "2026-05-09T14:50:58+00:00",
                    "notes": "ok",
                }
            ],
            "freshness_policy": {
                "stale_after_hours": 24,
                "review_after_hours": 168,
            },
            "catalog": {
                "openrouter_model_count": 100,
                "ranked_candidate_count": 2,
                "latest_openrouter_models": [],
                "discovery_alerts": [],
            },
            "models": [
                {
                    "id": "model-a",
                    "name": "Model A",
                    "provider": "Provider",
                    "provider_group": "Open",
                    "parameter_note": "10B",
                    "openrouter_id": "provider/model-a",
                    "intelligence": 85,
                    "value_index": 50,
                    "input_per_million": 0.1,
                    "output_per_million": 0.3,
                    "blended_per_million": 0.4,
                    "context_length": 1000,
                    "context": "1K",
                    "sources": [{"label": "OR", "url": "https://openrouter.ai/provider/model-a"}],
                },
                {
                    "id": "model-b",
                    "name": "Model B",
                    "provider": "Provider",
                    "provider_group": "Open",
                    "parameter_note": "20B",
                    "openrouter_id": "provider/model-b",
                    "intelligence": 90,
                    "value_index": 100,
                    "input_per_million": 0.5,
                    "output_per_million": 1.0,
                    "blended_per_million": 1.5,
                    "context_length": 2000,
                    "context": "2K",
                    "sources": [{"label": "OR", "url": "https://openrouter.ai/provider/model-b"}],
                },
            ],
            "rankings": {
                "intelligence": ["model-b", "model-a"],
                "value": ["model-a", "model-b"],
                "cheap": ["model-a", "model-b"],
                "value_index": {"model-a": 50, "model-b": 100},
            },
        }
        if overrides:
            for k, v in overrides.items():
                payload[k] = v
        return payload

    def test_valid_payload_passes(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        summary = validate_payload(payload)
        self.assertTrue(len(summary) > 0)

    def test_intelligence_range_validated(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        payload["models"][0]["intelligence"] = 150
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_value_index_range_validated(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        payload["models"][0]["value_index"] = -5
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_missing_field_detected(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        del payload["models"][0]["name"]
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_duplicate_model_id_detected(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        payload["models"].append(payload["models"][0].copy())
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_duplicate_ranking_entry_detected(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        payload["rankings"]["intelligence"] = ["model-a", "model-a"]
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_unknown_ranking_id_detected(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        payload["rankings"]["intelligence"] = ["model-a", "nonexistent"]
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_missing_cost_in_value_ranking_detected(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        payload["models"][0]["blended_per_million"] = None
        payload["rankings"]["value"] = ["model-a", "model-b"]
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_future_timestamp_detected(self):
        from scripts.validate_rankings import validate_payload
        from datetime import datetime, timezone, timedelta
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        payload = self._make_payload()
        payload["generated_at"] = future
        with self.assertRaises(ValueError):
            validate_payload(payload, max_age_hours=24)

    def test_staleness_check(self):
        from scripts.validate_rankings import validate_payload
        old = "2020-01-01T00:00:00+00:00"
        payload = self._make_payload()
        payload["generated_at"] = old
        with self.assertRaises(ValueError):
            validate_payload(payload, max_age_hours=24)

    def test_missing_openrouter_source_detected(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        payload["source_status"] = []
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_missing_catalog_fields_detected(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        del payload["catalog"]["openrouter_model_count"]
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_missing_freshness_policy_detected(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        del payload["freshness_policy"]
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_url_validation(self):
        from scripts.validate_rankings import validate_payload
        payload = self._make_payload()
        payload["models"][0]["sources"][0]["url"] = "not-a-url"
        with self.assertRaises(ValueError):
            validate_payload(payload)


if __name__ == "__main__":
    unittest.main()
