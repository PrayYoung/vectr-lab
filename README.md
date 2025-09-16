# vectr-lab

vectr-lab is a research-ready scaffold for running multi-asset portfolio backtests with [vectorbt](https://vectorbt.dev/). It ships with opinionated defaults for data ingestion, reusable strategy components, risk management, and iterative research workflows.

## Quick start

```bash
pip install -e .
vectr data pull
vectr backtest run
```

### One-stop runner

Prefer a single command? Use the bundled script:

```bash
python scripts/run_experiment.py \
  --universe config/defaults/universe.yaml \
  --strategy config/defaults/strategy.yaml \
  --risk config/defaults/risk.yaml

Add `--debug` for verbose diagnostics, `--json-logs` for machine-readable output, and repeatable `--trace TICKER` flags to dump per-ticker aligned arrays for deep dives (saved under `trace_<ticker>.parquet`).
```

Override any of those paths with your own YAML files; the script will download data (using cache if present), run the backtest, and emit artifacts plus a report under `./artifacts/runs/<timestamp>/`.

## Environment management with uv

If you prefer using [uv](https://github.com/astral-sh/uv) for dependency management, install `uv` and run:

```bash
# create a virtualenv under ./.venv and install the project (with dev extras)
./scripts/uv_env.sh bootstrap

# later sync dependencies or execute project commands inside the managed environment
./scripts/uv_env.sh install
./scripts/uv_env.sh test
./scripts/uv_env.sh cli backtest run --universe config/defaults/universe.yaml
```

The helper script wraps `uv venv`, `uv pip install`, and `uv run` to keep the environment reproducible.

See `config/defaults/*.yaml` for example universes, strategy parameters, and risk settings. Results are stored under `./artifacts` by default.

## Features

- YAML + Pydantic configuration models with sane defaults
- Local OHLCV cache backed by parquet files from Yahoo Finance
- Strategy interface with reusable indicators and signal generation
- Fixed-risk position sizing and portfolio-level risk rules
- vectorbt-based runner with metrics, plots, and parameter scans
- Rich-powered Typer CLI for data pulls, backtests, scans, and reports

## Caveats

- Yahoo Finance data may contain survivor bias and is subject to availability
- Timezone and session handling defaults to exchange data; confirm for your venue
- The scaffold assumes end-of-day signal execution; lower latencies require validation

## Roadmap

- Database persistence adapters (Supabase/Postgres)
- Live trading adapters for brokers such as Alpaca leveraging shared components
