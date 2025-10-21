# pickafresa_vision secrets and configuration

To keep API keys and non-public variables private, this folder supports environment-based configuration with optional `.env` files for local development.

## Quick start

1. Create a local `.env` by copying the example:
   - `cp pickafresa_vision/.env.example pickafresa_vision/.env`
2. Fill in your secrets in `pickafresa_vision/.env` (never commit this file).
3. Ensure dependencies are installed (in your virtualenv):
   - python-dotenv is listed in `requirements.txt`.
4. Run scripts; they will read from environment variables automatically.

## Required variables (examples)

- `ROBOFLOW_API_KEY`: Your Roboflow API token
- Optional defaults for convenience:
  - `ROBOFLOW_WORKSPACE`
  - `ROBOFLOW_PROJECT`
  - `ROBOFLOW_VERSION`

## How it works

- `pickafresa_vision/config.py` tries to load a `.env` from this folder (or the repo root) if present, using `python-dotenv`.
- Use `require_env("VAR_NAME")` to fetch required secrets with a clear error if missing.
- Use `get_env("VAR_NAME", default)` for optional values.
- The repo `.gitignore` ignores `.env` files so secrets are not committed.

## macOS zsh tips

- Temporarily export a secret for one command:
  - `ROBOFLOW_API_KEY=... python pickafresa_vision/roboflow/roboflow_model_upload.py`
- Persist in your shell for current session:
  - `export ROBOFLOW_API_KEY=...`
- Or use the `.env` file for local dev.

## CI/CD

- In GitHub Actions or other CI, add secrets in the platform's secret store (e.g., `ROBOFLOW_API_KEY`) and expose them as environment variables to jobs.

