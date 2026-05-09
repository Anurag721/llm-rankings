# LLM Signal — Frontier LLM Rankings

A static website for ranking large LLMs in three user-facing categories:

1. **Most intelligent** — quality-first frontier models.
2. **Cost-efficient intelligence** — strong models with sane production pricing.
3. **Cheap big models** — lowest-cost 100B+ / large MoE options.

## Run locally

```bash
cd /root/llm-rankings
python3 -m http.server 4173
```

Open: http://localhost:4173

## Automated ingestion pipeline

The site now reads generated data from:

```text
data/rankings.json
```

Regenerate it with:

```bash
cd /root/llm-rankings
python3 scripts/ingest_rankings.py
```

Use cached upstream payloads when offline:

```bash
python3 scripts/ingest_rankings.py --offline
```

`--offline` never reaches the network. It uses `data/.cache/openrouter-models.json` when present, then falls back to the committed `data/rankings.json` snapshot so local builds can still run after a fresh clone.

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Validate the committed payload:

```bash
python3 scripts/validate_rankings.py data/rankings.json --max-age-hours 48
```

GitHub Actions refreshes `data/rankings.json` from live sources every 6 hours and commits the result when it changes. CI also rejects stale generated data.

## Pipeline sources

- **OpenRouter API** — live model metadata, context windows, and normalized token prices.
- **Official provider pages** — curated source links and parameter notes for OpenAI, Anthropic, Google, xAI, DeepSeek, Qwen, Meta, Moonshot/Kimi, and Mistral.
- **Artificial Analysis** — public leaderboard page is fetched/cached and tracked in `source_status`.
- **LMArena** — public leaderboard page is fetched/cached and tracked in `source_status`.
- **OpenRouter rankings** — public rankings page is fetched/cached and tracked in `source_status`.

Benchmark sites often render leaderboards client-side or change private APIs. The adapter records source health today and is ready to blend machine-readable scores when a stable public feed is available.

## Data policy

- Scope is mainly 100B+ public-parameter models plus closed flagship frontier models with undisclosed parameter counts.
- Token costs are normalized as dollars per 1M input/output tokens.
- Blended cost = input $/M + output $/M.
- Intelligence score is currently a curated benchmark-consensus seed, blended with benchmark adapter scores when available.
- Value ranking filters to models with intelligence >= 83, then normalizes intelligence per blended dollar.
- Newly discovered OpenRouter models that look frontier-scale are shown as review alerts instead of being auto-ranked without a benchmark seed.
