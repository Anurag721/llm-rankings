#!/usr/bin/env python3
"""Automated data ingestion for the LLM Signal static site.

The pipeline is intentionally source-first:
- OpenRouter API provides normalized model metadata, context windows, and token pricing.
- Curated official provider links document model identity and parameter notes.
- Artificial Analysis and LMArena adapters probe/cache public benchmark pages and expose
  source status. If a public machine-readable feed is added later, plug it into
  collect_benchmark_signals() without changing the frontend contract.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
BENCHMARK_SOURCES = [
    {
        "id": "artificial-analysis",
        "label": "Artificial Analysis LLM leaderboards",
        "url": "https://artificialanalysis.ai/leaderboards/models",
        "type": "benchmark-index",
    },
    {
        "id": "lmarena",
        "label": "LMArena leaderboard",
        "url": "https://lmarena.ai/leaderboard",
        "type": "human-preference",
    },
    {
        "id": "openrouter-rankings",
        "label": "OpenRouter rankings",
        "url": "https://openrouter.ai/rankings",
        "type": "usage-community",
    },
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_json(url: str, timeout: int = 30) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "LLMSignalBot/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str, timeout: int = 20) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "LLMSignalBot/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def dollars_per_million(raw: str | int | float | None) -> float | None:
    if raw in (None, ""):
        return None
    return round(float(raw) * 1_000_000, 6)


def format_context(tokens: int | None) -> str:
    if not tokens:
        return "Unknown"
    if tokens >= 1_000_000:
        value = tokens / 1_000_000
        return f"{value:g}M"
    return f"{round(tokens / 1000):g}K"


def extract_openrouter_model(openrouter_payload: dict[str, Any], openrouter_id: str) -> dict[str, Any] | None:
    by_id = {model.get("id"): model for model in openrouter_payload.get("data", [])}
    model = by_id.get(openrouter_id)
    if not model:
        return None
    pricing = model.get("pricing") or {}
    input_cost = dollars_per_million(pricing.get("prompt"))
    output_cost = dollars_per_million(pricing.get("completion"))
    blended = None
    if input_cost is not None and output_cost is not None:
        blended = round(input_cost + output_cost, 6)
    return {
        "openrouter_id": model.get("id"),
        "openrouter_name": model.get("name"),
        "context_length": model.get("context_length"),
        "input_per_million": input_cost,
        "output_per_million": output_cost,
        "blended_per_million": blended,
        "knowledge_cutoff": model.get("knowledge_cutoff"),
        "sources": [
            {
                "label": "OpenRouter",
                "url": f"https://openrouter.ai/{openrouter_id}",
                "type": "pricing",
            }
        ],
    }


def collect_benchmark_signals(cache_dir: Path, offline: bool = False) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Probe public benchmark pages and return available source status.

    Public benchmark sites frequently render leaderboards client-side or protect internal APIs.
    This adapter records successful retrieval and can parse simple model-name mentions when
    present. The curated intelligence seed remains the fallback when no machine-readable score
    is available.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    signals: dict[str, float] = {}
    statuses: list[dict[str, Any]] = []
    for source in BENCHMARK_SOURCES:
        status = {**source, "status": "not-fetched", "fetched_at": None, "notes": ""}
        cache_path = cache_dir / f"{source['id']}.html"
        try:
            if offline and cache_path.exists():
                html = cache_path.read_text(encoding="utf-8", errors="replace")
                status["status"] = "cached"
            elif offline:
                html = ""
                status["status"] = "missing-cache"
            else:
                html = fetch_text(source["url"])
                cache_path.write_text(html, encoding="utf-8")
                status["status"] = "fetched"
            status["fetched_at"] = now_iso()
            status["notes"] = summarize_benchmark_page(html)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status["status"] = "error"
            status["fetched_at"] = now_iso()
            status["notes"] = str(exc)[:180]
        statuses.append(status)
    return signals, statuses


def summarize_benchmark_page(html: str) -> str:
    if not html:
        return "No page content available; using curated benchmark seed scores."
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "page fetched"
    return f"Fetched public page ({title}); no stable public score API assumed."


def intelligence_score(candidate: dict[str, Any], benchmark_signals: dict[str, float]) -> int:
    seed = float(candidate.get("intelligence_seed", 70))
    signal = benchmark_signals.get(candidate["id"])
    if signal is None:
        return int(round(seed))
    return int(round((seed * 0.55) + (signal * 0.45)))


def enrich_candidates(
    candidates: list[dict[str, Any]],
    openrouter_payload: dict[str, Any],
    benchmark_signals: dict[str, float],
) -> list[dict[str, Any]]:
    enriched = []
    for candidate in candidates:
        live = extract_openrouter_model(openrouter_payload, candidate["openrouter_id"])
        model = {
            "id": candidate["id"],
            "name": candidate["name"],
            "provider": candidate["provider"],
            "provider_group": candidate["provider_group"],
            "parameter_note": candidate["parameter_note"],
            "note": candidate.get("note", ""),
            "intelligence": intelligence_score(candidate, benchmark_signals),
            "sources": list(candidate.get("official_sources", [])),
        }
        if live:
            model.update(
                {
                    "openrouter_id": live["openrouter_id"],
                    "context_length": live["context_length"],
                    "context": format_context(live["context_length"]),
                    "input_per_million": live["input_per_million"],
                    "output_per_million": live["output_per_million"],
                    "blended_per_million": live["blended_per_million"],
                    "knowledge_cutoff": live["knowledge_cutoff"],
                }
            )
            model["sources"] = live["sources"] + model["sources"]
        else:
            model.update(
                {
                    "openrouter_id": candidate["openrouter_id"],
                    "context_length": None,
                    "context": "Unknown",
                    "input_per_million": None,
                    "output_per_million": None,
                    "blended_per_million": math.inf,
                    "knowledge_cutoff": None,
                }
            )
            model["sources"].insert(
                0,
                {
                    "label": "OpenRouter lookup missing",
                    "url": f"https://openrouter.ai/{candidate['openrouter_id']}",
                    "type": "pricing-warning",
                },
            )
        enriched.append(model)
    return enriched


def compute_rankings(models: list[dict[str, Any]]) -> dict[str, Any]:
    def cost(model: dict[str, Any]) -> float:
        value = model.get("blended_per_million")
        return float(value) if value is not None else math.inf

    ranked_by_value_raw = {
        model["id"]: model["intelligence"] / max(cost(model), 0.05)
        for model in models
        if math.isfinite(cost(model))
    }
    max_value = max(ranked_by_value_raw.values(), default=1)
    value_index = {
        model_id: int(round(raw / max_value * 100))
        for model_id, raw in ranked_by_value_raw.items()
    }
    for model in models:
        model["value_index"] = value_index.get(model["id"], 0)

    return {
        "intelligence": [m["id"] for m in sorted(models, key=lambda m: m["intelligence"], reverse=True)[:6]],
        "value": [
            m["id"]
            for m in sorted(
                [m for m in models if m["intelligence"] >= 83 and math.isfinite(cost(m))],
                key=lambda m: (m["value_index"], m["intelligence"]),
                reverse=True,
            )[:6]
        ],
        "cheap": [m["id"] for m in sorted([m for m in models if math.isfinite(cost(m))], key=cost)[:6]],
        "value_index": value_index,
    }


def build_payload(candidates_path: Path, offline: bool = False) -> dict[str, Any]:
    candidates = load_json(candidates_path)
    cache_dir = DATA_DIR / ".cache"
    openrouter_cache = cache_dir / "openrouter-models.json"
    if offline and openrouter_cache.exists():
        openrouter_payload = load_json(openrouter_cache)
    else:
        openrouter_payload = fetch_json(OPENROUTER_MODELS_URL)
        write_json(openrouter_cache, openrouter_payload)
    benchmark_signals, benchmark_statuses = collect_benchmark_signals(cache_dir, offline=offline)
    models = enrich_candidates(candidates, openrouter_payload, benchmark_signals)
    rankings = compute_rankings(models)
    return {
        "generated_at": now_iso(),
        "scope": "Large LLMs: public 100B+ parameter models plus closed frontier-scale models with undisclosed size.",
        "methodology": {
            "intelligence": "Curated benchmark-consensus seed, blended with machine-readable benchmark signals when available from adapters.",
            "cost": "OpenRouter prompt/completion pricing normalized to dollars per million tokens. Blended = input + output.",
            "value": "Normalized intelligence per blended dollar, with low-cost models capped by their intelligence score.",
        },
        "source_status": [
            {
                "id": "openrouter-models",
                "label": "OpenRouter model/pricing API",
                "url": OPENROUTER_MODELS_URL,
                "type": "pricing",
                "status": "fetched" if not offline else "cached",
                "fetched_at": now_iso(),
                "notes": f"{len(openrouter_payload.get('data', []))} models available in API payload.",
            },
            *benchmark_statuses,
        ],
        "models": models,
        "rankings": rankings,
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest LLM ranking data")
    parser.add_argument("--candidates", type=Path, default=DATA_DIR / "candidates.json")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "rankings.json")
    parser.add_argument("--offline", action="store_true", help="Use cached upstream payloads")
    args = parser.parse_args(argv)
    start = time.time()
    payload = build_payload(args.candidates, offline=args.offline)
    write_json(args.output, payload)
    print(f"Wrote {args.output} with {len(payload['models'])} models in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
