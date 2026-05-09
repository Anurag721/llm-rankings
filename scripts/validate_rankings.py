#!/usr/bin/env python3
"""Validate the generated ranking payload used by the static site.

Performs structural, type, range, and cross-reference checks on rankings.json
so the frontend can render confidently without additional defensive code.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
log = logging.getLogger("validate_rankings")

# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

VALID_FIELDS = {
    "id", "name", "provider", "provider_group", "parameter_note", "openrouter_id",
    "context", "context_length", "intelligence", "value_index",
    "input_per_million", "output_per_million", "blended_per_million",
    "knowledge_cutoff", "created_at", "canonical_slug", "note", "sources",
}


def load_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    return payload


def parse_iso(value: str, field: str) -> datetime:
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_number(value: Any, field: str, *, allow_null: bool = False,
                    min_val: float | None = None, max_val: float | None = None) -> None:
    if value is None and allow_null:
        return
    require(isinstance(value, (int, float)), f"{field} must be numeric")
    require(math.isfinite(float(value)), f"{field} must be finite")
    require(float(value) >= 0, f"{field} must be non-negative")
    if min_val is not None:
        require(float(value) >= min_val, f"{field} must be >= {min_val}")
    if max_val is not None:
        require(float(value) <= max_val, f"{field} must be <= {max_val}")


def validate_url(value: Any, field: str) -> None:
    require(isinstance(value, str) and value.startswith(("http://", "https://")),
            f"{field} must be an HTTPS/HTTP URL")


def validate_payload(payload: dict[str, Any], max_age_hours: float | None = None) -> list[str]:
    # --- Top-level timestamps ---
    generated_at = parse_iso(payload.get("generated_at", ""), "generated_at")
    age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
    require(age_hours > -1, "generated_at is unexpectedly in the future")
    if max_age_hours is not None:
        require(age_hours <= max_age_hours, f"payload is stale: {age_hours:.1f}h old > {max_age_hours:g}h")

    # --- Models array ---
    models = payload.get("models")
    require(isinstance(models, list) and models, "models must be a non-empty list")
    model_ids = [model.get("id") for model in models if isinstance(model, dict)]
    require(len(model_ids) == len(models), "every model must be an object with an id")
    require(len(model_ids) == len(set(model_ids)), "model ids must be unique")
    known_ids = set(model_ids)

    for model in models:
        model_id = model["id"]
        # Required fields
        for field in ("name", "provider", "provider_group", "parameter_note", "openrouter_id"):
            require(bool(model.get(field)), f"{model_id} missing {field}")
        # Numeric ranges
        validate_number(model.get("intelligence"), f"{model_id}.intelligence", min_val=0, max_val=100)
        validate_number(model.get("value_index"), f"{model_id}.value_index", min_val=0, max_val=100)
        validate_number(model.get("input_per_million"), f"{model_id}.input_per_million", allow_null=True)
        validate_number(model.get("output_per_million"), f"{model_id}.output_per_million", allow_null=True)
        validate_number(model.get("blended_per_million"), f"{model_id}.blended_per_million", allow_null=True)
        # Context length
        ctx = model.get("context_length")
        if ctx is not None:
            require(isinstance(ctx, int) and ctx > 0, f"{model_id}.context_length must be a positive integer")
        # Sources array
        sources = model.get("sources")
        if sources is not None:
            require(isinstance(sources, list), f"{model_id}.sources must be a list")
            for i, src in enumerate(sources):
                require(isinstance(src, dict), f"{model_id}.sources[{i}] must be an object")
                if src.get("url"):
                    validate_url(src["url"], f"{model_id}.sources[{i}].url")
        # Unknown fields (catch typos)
        extra = set(model.keys()) - VALID_FIELDS
        if extra:
            log.warning("%s has unexpected fields: %s", model_id, sorted(extra))

    # --- Rankings ---
    rankings = payload.get("rankings")
    require(isinstance(rankings, dict), "rankings must be an object")
    for ranking_name in ("intelligence", "value", "cheap"):
        ids = rankings.get(ranking_name)
        require(isinstance(ids, list) and ids, f"rankings.{ranking_name} must be a non-empty list")
        unknown = sorted(set(ids) - known_ids)
        require(not unknown, f"rankings.{ranking_name} references unknown ids: {', '.join(unknown)}")
        # Check no duplicates in ranking list
        dupes = [x for x in ids if ids.count(x) > 1]
        require(not dupes, f"rankings.{ranking_name} has duplicate entries: {set(dupes)}")
        # Cost-required rankings
        if ranking_name in {"value", "cheap"}:
            missing_cost = [
                mid for mid in ids
                if next(m for m in models if m["id"] == mid).get("blended_per_million") is None
            ]
            require(not missing_cost,
                    f"rankings.{ranking_name} includes models without cost: {', '.join(missing_cost)}")

    # --- Value index consistency ---
    value_index = rankings.get("value_index")
    require(isinstance(value_index, dict), "rankings.value_index must be an object")
    require(set(value_index) == known_ids,
            "rankings.value_index must include every model id exactly once")
    for mid, vi in value_index.items():
        validate_number(vi, f"value_index[{mid}]", min_val=0, max_val=100)

    # --- Source status ---
    source_status = payload.get("source_status")
    require(isinstance(source_status, list) and source_status, "source_status must be a non-empty list")
    source_ids = {}
    for src in source_status:
        require(isinstance(src, dict), "each source_status entry must be an object")
        sid = src.get("id")
        require(bool(sid), "source_status entry must have a non-empty id")
        require(sid not in source_ids, f"source_status has duplicate id: {sid}")
        source_ids[sid] = src
        if src.get("url"):
            validate_url(src["url"], f"source_status[{sid}].url")
    require("openrouter-models" in source_ids, "source_status must include openrouter-models")
    require(
        source_ids["openrouter-models"].get("status") not in {"error", "missing-cache"},
        "openrouter model source is not usable",
    )

    # --- Catalog ---
    catalog = payload.get("catalog")
    require(isinstance(catalog, dict), "catalog metadata must be present")
    require(isinstance(catalog.get("openrouter_model_count"), int),
            "catalog.openrouter_model_count must be an integer")
    require(isinstance(catalog.get("ranked_candidate_count"), int),
            "catalog.ranked_candidate_count must be an integer")
    require(isinstance(catalog.get("latest_openrouter_models"), list),
            "catalog.latest_openrouter_models must be present")
    require(isinstance(catalog.get("discovery_alerts"), list),
            "catalog.discovery_alerts must be present")

    # --- Freshness policy ---
    fp = payload.get("freshness_policy")
    require(isinstance(fp, dict), "freshness_policy must be present")
    validate_number(fp.get("stale_after_hours"), "freshness_policy.stale_after_hours")
    validate_number(fp.get("review_after_hours"), "freshness_policy.review_after_hours")

    return [
        f"models={len(models)}",
        f"sources={len(source_status)}",
        f"age_hours={age_hours:.2f}",
        f"discovery_alerts={len(catalog.get('discovery_alerts', []))}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated ranking payload")
    parser.add_argument("path", type=Path, help="Path to rankings.json")
    parser.add_argument("--max-age-hours", type=float, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        summary = validate_payload(load_payload(args.path), max_age_hours=args.max_age_hours)
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        log.error("Invalid rankings payload: %s", exc)
        return 1

    log.info("Valid rankings payload: %s", ", ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
