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

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

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
