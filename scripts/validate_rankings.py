#!/usr/bin/env python3
"""Validate the generated ranking payload used by the static site."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def validate_number(value: Any, field: str, *, allow_null: bool = False) -> None:
    if value is None and allow_null:
        return
    require(isinstance(value, (int, float)), f"{field} must be numeric")
    require(math.isfinite(float(value)), f"{field} must be finite")
    require(float(value) >= 0, f"{field} must be non-negative")


def validate_payload(payload: dict[str, Any], max_age_hours: float | None = None) -> list[str]:
    generated_at = parse_iso(payload.get("generated_at", ""), "generated_at")
    age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
    require(age_hours > -1, "generated_at is unexpectedly in the future")
    if max_age_hours is not None:
        require(age_hours <= max_age_hours, f"payload is stale: {age_hours:.1f}h old > {max_age_hours:g}h")

    models = payload.get("models")
    require(isinstance(models, list) and models, "models must be a non-empty list")
    model_ids = [model.get("id") for model in models if isinstance(model, dict)]
    require(len(model_ids) == len(models), "every model must be an object with an id")
    require(len(model_ids) == len(set(model_ids)), "model ids must be unique")
    known_ids = set(model_ids)

    for model in models:
        model_id = model["id"]
        for field in ("name", "provider", "provider_group", "parameter_note", "openrouter_id"):
            require(bool(model.get(field)), f"{model_id} missing {field}")
        validate_number(model.get("intelligence"), f"{model_id}.intelligence")
        validate_number(model.get("input_per_million"), f"{model_id}.input_per_million", allow_null=True)
        validate_number(model.get("output_per_million"), f"{model_id}.output_per_million", allow_null=True)
        validate_number(model.get("blended_per_million"), f"{model_id}.blended_per_million", allow_null=True)
        validate_number(model.get("value_index"), f"{model_id}.value_index")

    rankings = payload.get("rankings")
    require(isinstance(rankings, dict), "rankings must be an object")
    for ranking_name in ("intelligence", "value", "cheap"):
        ids = rankings.get(ranking_name)
        require(isinstance(ids, list) and ids, f"rankings.{ranking_name} must be a non-empty list")
        unknown = sorted(set(ids) - known_ids)
        require(not unknown, f"rankings.{ranking_name} references unknown ids: {', '.join(unknown)}")
        if ranking_name in {"value", "cheap"}:
            missing_cost = [model_id for model_id in ids if next(m for m in models if m["id"] == model_id).get("blended_per_million") is None]
            require(not missing_cost, f"rankings.{ranking_name} includes models without cost: {', '.join(missing_cost)}")

    value_index = rankings.get("value_index")
    require(isinstance(value_index, dict), "rankings.value_index must be an object")
    require(set(value_index) == known_ids, "rankings.value_index must include every model id exactly once")

    source_status = payload.get("source_status")
    require(isinstance(source_status, list) and source_status, "source_status must be a non-empty list")
    source_ids = {source.get("id"): source for source in source_status if isinstance(source, dict)}
    require("openrouter-models" in source_ids, "source_status must include openrouter-models")
    require(
        source_ids["openrouter-models"].get("status") not in {"error", "missing-cache"},
        "openrouter model source is not usable",
    )

    catalog = payload.get("catalog")
    require(isinstance(catalog, dict), "catalog metadata must be present")
    require(isinstance(catalog.get("latest_openrouter_models"), list), "catalog.latest_openrouter_models must be present")
    require(isinstance(catalog.get("discovery_alerts"), list), "catalog.discovery_alerts must be present")

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
    args = parser.parse_args(argv)

    try:
        summary = validate_payload(load_payload(args.path), max_age_hours=args.max_age_hours)
    except Exception as exc:
        print(f"Invalid rankings payload: {exc}", file=sys.stderr)
        return 1

    print(f"Valid rankings payload: {', '.join(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
