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

  - `ROBOFLOW_WORKSPACE`
  - `ROBOFLOW_PROJECT`
  - `ROBOFLOW_VERSION`

## How it works


## macOS zsh tips

  - `ROBOFLOW_API_KEY=... python pickafresa_vision/roboflow/roboflow_model_upload.py`
  - `export ROBOFLOW_API_KEY=...`

## CI/CD

- In GitHub Actions or other CI, add secrets in the platform's secret store (e.g., `ROBOFLOW_API_KEY`) and expose them as environment variables to jobs.

## RealSense verification tools (macOS stability)

These scripts live in `pickafresa_vision/realsense_testing/`:

- `realsense_verify_color.py`
- `realsense_verify_depth.py`
- `realsense_verify_full.py` (color + depth)

On macOS (Apple Silicon), librealsense pipeline start/stop can occasionally crash due to device-hub power toggles. Our verifiers default to DIRECT sensor mode on macOS, which opens and starts sensors directly with frame queues. You can adjust behavior via environment variables:

- `REALSENSE_COLOR_USE_PIPELINE=1`
  - Force the color verifier to use the pipeline fallback instead of DIRECT mode.

- `REALSENSE_DEPTH_USE_PIPELINE=1`
  - Force the depth verifier to use the pipeline fallback instead of DIRECT mode.

- `REALSENSE_FULL_USE_PIPELINE=1`
  - Force the full verifier (color+depth) to use the pipeline fallback instead of DIRECT mode.

- `REALSENSE_FULL_SWEEP=1`
  - When cached working profiles exist, bypass the cache and run a fresh verification sweep. If unset, the scripts will prefer cached results and, in interactive terminals, prompt whether to run a full sweep.

Examples (zsh):

```zsh
# Force pipeline for color script
REALSENSE_COLOR_USE_PIPELINE=1 ./realsense_venv_sudo pickafresa_vision/realsense_testing/realsense_verify_color.py

# Force pipeline for full script and run a full sweep instead of cache
REALSENSE_FULL_USE_PIPELINE=1 REALSENSE_FULL_SWEEP=1 ./realsense_venv_sudo pickafresa_vision/realsense_testing/realsense_verify_full.py
```

Logs for troubleshooting are written to:

- `pickafresa_vision/logs/realsense_color_verify.log`
- `pickafresa_vision/logs/realsense_depth_verify.log`
- `pickafresa_vision/logs/realsense_full_verify.log`

Notes:

- DIRECT mode is the default only on macOS. On other platforms, the pipeline path is used.
- The cache stores known-good profiles per camera serial and speeds up subsequent runs. You can still validate cached profiles explicitly via the library API helpers.

