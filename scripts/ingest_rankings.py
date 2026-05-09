#!/usr/bin/env python3
"""Automated data ingestion for the LLM Signal static site.

The pipeline is intentionally source-first:
- OpenRouter API provides normalized model metadata, context windows, and token pricing.
- Curated official provider links document model identity and parameter notes.
- Artificial Analysis and LMArena adapters probe/cache public benchmark pages and expose
  source status. If a public machine-readable feed is added later, plug it into
  collect_benchmark_signals() without changing the frontend contract.

Configuration is loaded from config.yaml when present; CLI flags override it.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger("ingest_rankings")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
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
    "benchmark_sources": [
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
    ],
    "frontier_detection": {
        "parameter_pattern": r"\b(?:[1-9]\d{2,}(?:\.\d+)?\s*B|[1-9](?:\.\d+)?\s*T)\b",
        "signal_pattern": (
            r"\b(frontier|flagship|reasoning|agentic|coding agent"
            r"|foundation model|large language model)\b"
        ),
        "providers": ["anthropic", "google", "openai", "x-ai"],
    },
    "paths": {
        "candidates": "data/candidates.json",
        "output": "data/rankings.json",
        "cache_dir": "data/.cache",
    },
}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load config from YAML, falling back to defaults."""
    cfg = _deep_copy(DEFAULT_CONFIG)
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if config_path.exists():
        if not HAS_YAML:
            log.warning(
                "PyYAML not installed; cannot load %s. Using built-in defaults.", config_path
            )
            return cfg
        with open(config_path) as fh:
            user_cfg = yaml.safe_load(fh) or {}
        _deep_merge(cfg, user_cfg)
        log.info("Loaded config from %s", config_path)
    else:
        log.info("No config file at %s; using built-in defaults.", config_path)
    return cfg


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(i) for i in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = _deep_copy(v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def fetch_with_retry(url: str, *, timeout: int = 30, max_retries: int = 3,
                     backoff: int = 2, headers: dict[str, str] | None = None) -> bytes:
    """Fetch a URL with exponential backoff retries on transient errors."""
    attempt_headers = headers or {}
    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(url, headers=attempt_headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == max_retries:
                log.error("Failed to fetch %s after %d attempts: %s", url, max_retries, exc)
                raise
            wait = backoff * (2 ** (attempt - 1))
            log.warning(
                "Attempt %d/%d for %s failed (%s); retrying in %ds",
                attempt, max_retries, url, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Unreachable: fetch_with_retry exited loop for {url}")


def fetch_json(url: str, *, cfg: dict[str, Any]) -> Any:
    timeout = cfg["pipeline"].get("fetch_timeout_seconds", 30)
    max_retries = cfg["pipeline"].get("max_retries", 3)
    backoff = cfg["pipeline"].get("retry_backoff_seconds", 2)
    user_agent = cfg["pipeline"].get("user_agent", "LLMSignalBot/1.0")
    data = fetch_with_retry(
        url, timeout=timeout, max_retries=max_retries, backoff=backoff,
        headers={"User-Agent": user_agent},
    )
    return json.loads(data.decode("utf-8"))


def fetch_text(url: str, *, cfg: dict[str, Any]) -> str:
    timeout = cfg["pipeline"].get("text_fetch_timeout_seconds", 20)
    max_retries = cfg["pipeline"].get("max_retries", 3)
    backoff = cfg["pipeline"].get("retry_backoff_seconds", 2)
    user_agent = cfg["pipeline"].get("user_agent", "LLMSignalBot/1.0")
    data = fetch_with_retry(
        url, timeout=timeout, max_retries=max_retries, backoff=backoff,
        headers={"User-Agent": user_agent},
    )
    return data.decode("utf-8", errors="replace")


def dollars_per_million(raw: str | int | float | None) -> float | None:
    """Convert per-token cost to dollars per million tokens."""
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


def unix_to_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).replace(microsecond=0).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# OpenRouter extraction
# ---------------------------------------------------------------------------

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
        "canonical_slug": model.get("canonical_slug"),
        "context_length": model.get("context_length"),
        "input_per_million": input_cost,
        "output_per_million": output_cost,
        "blended_per_million": blended,
        "knowledge_cutoff": model.get("knowledge_cutoff"),
        "created_at": unix_to_iso(model.get("created")),
        "sources": [
            {
                "label": "OpenRouter",
                "url": f"https://openrouter.ai/{openrouter_id}",
                "type": "pricing",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Benchmark adapters
# ---------------------------------------------------------------------------

def collect_benchmark_signals(
    cache_dir: Path, *, cfg: dict[str, Any], offline: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Probe public benchmark pages and return available source status."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    signals: dict[str, float] = {}
    statuses: list[dict[str, Any]] = []
    sources = cfg.get("benchmark_sources", DEFAULT_CONFIG["benchmark_sources"])
    for source in sources:
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
                html = fetch_text(source["url"], cfg=cfg)
                cache_path.write_text(html, encoding="utf-8")
                status["status"] = "fetched"
            status["fetched_at"] = now_iso()
            status["notes"] = summarize_benchmark_page(html)
            log.info("Benchmark source '%s': %s", source["id"], status["status"])
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status["status"] = "error"
            status["fetched_at"] = now_iso()
            status["notes"] = str(exc)[:180]
            log.warning("Benchmark source '%s' failed: %s", source["id"], exc)
        statuses.append(status)
    return signals, statuses


def summarize_benchmark_page(html: str) -> str:
    if not html:
        return "No page content available; using curated benchmark seed scores."
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = (
        html_lib.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip())
        if title_match
        else "page fetched"
    )
    return f"Fetched public page ({title}); no stable public score API assumed."


# ---------------------------------------------------------------------------
# Scoring & ranking
# ---------------------------------------------------------------------------

def intelligence_score(
    candidate: dict[str, Any],
    benchmark_signals: dict[str, float],
    *,
    blend_ratio: float = 0.55,
    signal_ratio: float = 0.45,
) -> int:
    seed = float(candidate.get("intelligence_seed", 70))
    signal = benchmark_signals.get(candidate["id"])
    if signal is None:
        return int(round(seed))
    return int(round((seed * blend_ratio) + (signal * signal_ratio)))


def enrich_candidates(
    candidates: list[dict[str, Any]],
    openrouter_payload: dict[str, Any],
    benchmark_signals: dict[str, float],
) -> list[dict[str, Any]]:
    enriched = []
    for candidate in candidates:
        live = extract_openrouter_model(openrouter_payload, candidate["openrouter_id"])
        model: dict[str, Any] = {
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
                    "created_at": live["created_at"],
                    "canonical_slug": live["canonical_slug"],
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
                    "blended_per_million": None,
                    "knowledge_cutoff": None,
                    "created_at": None,
                    "canonical_slug": None,
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


def compute_rankings(
    models: list[dict[str, Any]],
    *,
    intelligence_cutoff: int = 83,
    max_results: int = 6,
    min_cost: float = 0.05,
) -> dict[str, Any]:
    def cost(model: dict[str, Any]) -> float:
        value = model.get("blended_per_million")
        return float(value) if value is not None else math.inf

    value_candidates = [
        model
        for model in models
        if model["intelligence"] >= intelligence_cutoff and math.isfinite(cost(model))
    ]
    ranked_by_value_raw = {
        model["id"]: model["intelligence"] / max(cost(model), min_cost)
        for model in value_candidates
    }
    max_value = max(ranked_by_value_raw.values(), default=1)
    value_index = {model["id"]: 0 for model in models}
    value_index.update(
        {
            model_id: int(round(raw / max_value * 100))
            for model_id, raw in ranked_by_value_raw.items()
        }
    )
    for model in models:
        model["value_index"] = value_index.get(model["id"], 0)

    return {
        "intelligence": [
            m["id"] for m in sorted(models, key=lambda m: m["intelligence"], reverse=True)[:max_results]
        ],
        "value": [
            m["id"]
            for m in sorted(
                value_candidates,
                key=lambda m: (m["value_index"], m["intelligence"]),
                reverse=True,
            )[:max_results]
        ],
        "cheap": [
            m["id"]
            for m in sorted(
                [m for m in models if math.isfinite(cost(m))], key=cost
            )[:max_results]
        ],
        "value_index": value_index,
    }


# ---------------------------------------------------------------------------
# Catalog & discovery
# ---------------------------------------------------------------------------

def is_frontier_discovery(
    model: dict[str, Any],
    *,
    param_re: re.Pattern[str],
    signal_re: re.Pattern[str],
    frontier_providers: set[str],
) -> bool:
    openrouter_id = model.get("id") or ""
    provider = openrouter_id.split("/", 1)[0]
    text = " ".join(str(model.get(key) or "") for key in ("id", "name", "description"))
    return bool(
        param_re.search(text)
        or (provider in frontier_providers and signal_re.search(text))
    )


def summarize_openrouter_model(model: dict[str, Any]) -> dict[str, Any]:
    pricing = model.get("pricing") or {}
    input_cost = dollars_per_million(pricing.get("prompt"))
    output_cost = dollars_per_million(pricing.get("completion"))
    blended = None
    if input_cost is not None and output_cost is not None:
        blended = round(input_cost + output_cost, 6)
    openrouter_id = model.get("id") or ""
    return {
        "id": openrouter_id,
        "name": model.get("name") or openrouter_id,
        "created_at": unix_to_iso(model.get("created")),
        "context_length": model.get("context_length"),
        "context": format_context(model.get("context_length")),
        "input_per_million": input_cost,
        "output_per_million": output_cost,
        "blended_per_million": blended,
        "url": f"https://openrouter.ai/{openrouter_id}" if openrouter_id else None,
    }


def build_catalog_metadata(
    openrouter_payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    discovery_limit: int = 12,
    param_re: re.Pattern[str],
    signal_re: re.Pattern[str],
    frontier_providers: set[str],
) -> dict[str, Any]:
    openrouter_models = [
        model for model in openrouter_payload.get("data", []) if model.get("id")
    ]
    sorted_models = sorted(
        openrouter_models, key=lambda m: int(m.get("created") or 0), reverse=True
    )
    candidate_openrouter_ids = {candidate["openrouter_id"] for candidate in candidates}
    discovery_alerts = [
        summarize_openrouter_model(model)
        for model in sorted_models
        if model.get("id") not in candidate_openrouter_ids
        and is_frontier_discovery(
            model,
            param_re=param_re,
            signal_re=signal_re,
            frontier_providers=frontier_providers,
        )
    ][:discovery_limit]
    return {
        "openrouter_model_count": len(openrouter_models),
        "ranked_candidate_count": len(candidates),
        "latest_openrouter_models": [
            summarize_openrouter_model(model) for model in sorted_models[:discovery_limit]
        ],
        "discovery_alerts": discovery_alerts,
    }


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def build_payload(
    candidates_path: Path,
    *,
    cfg: dict[str, Any],
    offline: bool = False,
) -> dict[str, Any]:
    pipeline = cfg["pipeline"]
    candidates = load_json(candidates_path)
    cache_dir = ROOT / cfg["paths"]["cache_dir"]
    openrouter_payload, openrouter_status = load_openrouter_payload(cache_dir, cfg=cfg, offline=offline)
    benchmark_signals, benchmark_statuses = collect_benchmark_signals(
        cache_dir, cfg=cfg, offline=offline
    )

    models = enrich_candidates(candidates, openrouter_payload, benchmark_signals)
    rankings = compute_rankings(
        models,
        intelligence_cutoff=pipeline.get("intelligence_cutoff_for_value", 83),
        max_results=pipeline.get("max_ranked_results", 6),
        min_cost=pipeline.get("min_cost_for_value", 0.05),
    )

    fd = cfg.get("frontier_detection", DEFAULT_CONFIG["frontier_detection"])
    param_re = re.compile(fd["parameter_pattern"], re.I)
    signal_re = re.compile(fd["signal_pattern"], re.I)
    frontier_providers = set(fd.get("providers", []))

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
                "url": cfg["pipeline"]["openrouter_url"],
                "type": "pricing",
                "status": openrouter_status["status"],
                "fetched_at": now_iso(),
                "notes": openrouter_status["notes"],
            },
            *benchmark_statuses,
        ],
        "freshness_policy": {
            "stale_after_hours": pipeline.get("stale_after_hours", 24),
            "review_after_hours": pipeline.get("review_after_hours", 168),
        },
        "catalog": build_catalog_metadata(
            openrouter_payload,
            candidates,
            discovery_limit=pipeline.get("discovery_limit", 12),
            param_re=param_re,
            signal_re=signal_re,
            frontier_providers=frontier_providers,
        ),
        "models": models,
        "rankings": rankings,
    }


# ---------------------------------------------------------------------------
# OpenRouter payload loading
# ---------------------------------------------------------------------------

def openrouter_payload_from_rankings(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rankings_payload = load_json(path)
    models = []
    for model in rankings_payload.get("models", []):
        openrouter_id = model.get("openrouter_id")
        if not openrouter_id:
            continue
        input_cost = model.get("input_per_million")
        output_cost = model.get("output_per_million")
        models.append(
            {
                "id": openrouter_id,
                "canonical_slug": model.get("canonical_slug") or openrouter_id,
                "name": model.get("openrouter_name") or model.get("name"),
                "context_length": model.get("context_length"),
                "pricing": {
                    "prompt": (
                        "" if input_cost is None else str(float(input_cost) / 1_000_000)
                    ),
                    "completion": (
                        "" if output_cost is None else str(float(output_cost) / 1_000_000)
                    ),
                },
                "knowledge_cutoff": model.get("knowledge_cutoff"),
                "created": None,
            }
        )
    if not models:
        return None
    return {
        "data": models,
        "_fallback_from_rankings": {
            "path": str(path),
            "generated_at": rankings_payload.get("generated_at"),
        },
    }


def load_openrouter_payload(
    cache_dir: Path, *, cfg: dict[str, Any], offline: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    openrouter_cache = cache_dir / "openrouter-models.json"
    if offline and openrouter_cache.exists():
        payload = load_json(openrouter_cache)
        return payload, {
            "status": "cached",
            "notes": f"{len(payload.get('data', []))} models loaded from local OpenRouter cache.",
        }
    if offline:
        fallback = openrouter_payload_from_rankings(
            ROOT / cfg["paths"]["output"]
        )
        if fallback:
            fallback_info = fallback.get("_fallback_from_rankings", {})
            return fallback, {
                "status": "from-generated",
                "notes": (
                    f"{len(fallback.get('data', []))} models reconstructed from committed rankings "
                    f"generated at {fallback_info.get('generated_at') or 'unknown time'}."
                ),
            }
        raise FileNotFoundError(
            f"Offline mode needs {openrouter_cache} or {cfg['paths']['output']}; "
            "run without --offline once to refresh the cache."
        )

    payload = fetch_json(cfg["pipeline"]["openrouter_url"], cfg=cfg)
    write_json(openrouter_cache, payload)
    return payload, {
        "status": "fetched",
        "notes": f"{len(payload.get('data', []))} models available in API payload.",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest LLM ranking data for the LLM Signal static site."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: config.yaml in project root)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Path to candidates.json (default from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output rankings.json (default from config)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached upstream payloads instead of fetching",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load_config(args.config)

    candidates_path = args.candidates or Path(cfg["paths"]["candidates"])
    output_path = args.output or Path(cfg["paths"]["output"])

    log.info("Starting ingestion: candidates=%s output=%s offline=%s", candidates_path, output_path, args.offline)
    start = time.time()
    payload = build_payload(candidates_path, cfg=cfg, offline=args.offline)
    write_json(output_path, payload)
    elapsed = time.time() - start
    log.info(
        "Wrote %s with %d models in %.1fs",
        output_path,
        len(payload["models"]),
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
