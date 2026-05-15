# GLDIS Configurations

This directory contains environment-specific configuration profiles.

## Files

| File | Purpose |
|------|---------|
| `settings.yaml` | Default settings with documentation |
| `dev.yaml` | Local development overrides |
| `prod.yaml` | Production overrides (use with PostgreSQL) |

## Usage

Settings are loaded in priority order:
1. Environment variables (highest priority)
2. `.env` file in project root
3. Default values in `core/config.py`

YAML files in this directory are **documentation/reference only** — the system
reads from environment variables. Use them as a reference when setting up
new deployment environments.
