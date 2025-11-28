"""
Upload local model weights from pickafresa_vision/models to a Roboflow project version.

Credentials and default IDs are read from environment variables if present:
  - ROBOFLOW_API_KEY (will prompt if missing)
  - ROBOFLOW_WORKSPACE (optional; will auto-detect from API key if missing)
  - ROBOFLOW_PROJECT (optional; will prompt if missing)
  - ROBOFLOW_VERSION (optional; will prompt if missing)

If IDs are not provided via env, the script will prompt for them in the CLI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roboflow import Roboflow

try:
    from pickafresa_vision.configs.config import get_env
except ImportError:
    # Fallback: read directly from os if config.py not available
    def get_env(name: str, default: str | None = None) -> str | None:
        val = os.getenv(name)
        if not val or val.strip() == "":
            return default
        return val


SUPPORTED_WEIGHT_EXTS = (
	".pt",
	".pth",
	".onnx",
	".tflite",
	".engine",
	".pb",
	".ckpt",
	".zip",
)


def get_api_key() -> str:
	"""Get Roboflow API key from environment or prompt user."""
	api_key = get_env("ROBOFLOW_API_KEY")
	if api_key:
		return api_key
	
	print("\nRoboflow API key not found in environment.")
	print("You can find your API key at: https://app.roboflow.com/settings/api")
	while True:
		api_key = input("\nEnter your Roboflow API key: ").strip()
		if api_key:
			return api_key
		print("API key cannot be empty. Please try again.")


def get_roboflow_client() -> Roboflow:
	api_key = get_api_key()
	return Roboflow(api_key=api_key)


def list_weight_files(models_dir: Path) -> List[Path]:
	if not models_dir.exists():
		return []
	files: List[Path] = []
	for p in sorted(models_dir.iterdir()):
		if p.is_file() and p.suffix.lower() in SUPPORTED_WEIGHT_EXTS:
			files.append(p)
	return files


def prompt_select_file(files: List[Path]) -> Optional[Path]:
	if not files:
		return None
	print("Select a weights file to upload:")
	for i, f in enumerate(files, start=1):
		print(f"  [{i}] {f.name}")
	while True:
		choice = input(f"Enter a number (1-{len(files)}): ").strip()
		if not choice.isdigit():
			print("Please enter a valid number.")
			continue
		idx = int(choice)
		if 1 <= idx <= len(files):
			return files[idx - 1]
		print("Choice out of range. Try again.")


def prompt_model_type() -> str:
	"""Prompt user for model type/format with examples."""
	model_type = get_env("ROBOFLOW_MODEL_TYPE")
	if model_type:
		return model_type
	
	print("\nEnter the model type/format for this weights file.")
	print("Examples:")
	print("  - yolov11 (or yolov11n, yolov11s, yolov11m, yolov11l, yolov11x)")
	print("  - yolov8 (or yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
	print("  - yolov5, yolov7, yolov9, yolov10")
	print("  - ultralytics (generic)")
	print("  - tensorflow, pytorch, onnx, tflite")
	print("  - custom (for custom architectures)")
	
	while True:
		model_type = input("\nModel type: ").strip()
		if model_type:
			return model_type
		print("Model type cannot be empty. Please enter a value.")


def resolve_project_ids(rf: Roboflow) -> tuple[str, str, str]:
	"""Resolve workspace, project, and version IDs, auto-detecting workspace from API."""
	workspace = get_env("ROBOFLOW_WORKSPACE")
	project = get_env("ROBOFLOW_PROJECT")
	version = get_env("ROBOFLOW_VERSION")

	# Auto-detect workspace if not provided
	if not workspace:
		try:
			# Try to list workspaces using the API
			# The Roboflow object has a .workspace() method, but we need the workspace ID
			# Try to get workspace from the API via listing projects
			print("\nFetching your Roboflow workspaces...")
			
			# Use the internal API to get workspace list
			if hasattr(rf, 'workspace'):
				# Try to get default workspace
				# Roboflow API structure: rf.__dict__ might have useful info
				import requests
				api_key = rf.api_key
				response = requests.get(
					"https://api.roboflow.com/",
					params={"api_key": api_key}
				)
				
				if response.status_code == 200:
					data = response.json()
					workspaces = data.get("workspace", [])
					
					if isinstance(workspaces, list) and len(workspaces) > 0:
						if len(workspaces) == 1:
							workspace = workspaces[0]["url"]
							print(f"Using workspace: {workspace}")
						else:
							print("\nAvailable workspaces:")
							for i, ws in enumerate(workspaces, start=1):
								print(f"  [{i}] {ws.get('name', 'Unknown')} ({ws['url']})")
							while True:
								choice = input(f"Select workspace (1-{len(workspaces)}): ").strip()
								if choice.isdigit() and 1 <= int(choice) <= len(workspaces):
									workspace = workspaces[int(choice) - 1]["url"]
									break
								print("Invalid selection. Try again.")
					elif isinstance(workspaces, str):
						# Single workspace returned as string
						workspace = workspaces
						print(f"Using workspace: {workspace}")
				
				if not workspace:
					raise ValueError("Could not fetch workspaces from API")
					
		except Exception as e:
			print(f"\nCould not auto-detect workspace: {e}")
			workspace = input("Enter Roboflow workspace ID manually: ").strip()
	
	if not project:
		project = input("Enter Roboflow project ID: ").strip()
	
	while not version:
		version = input("Enter Roboflow version number: ").strip()
		if not version:
			print("Version cannot be empty.")
			continue
		if not version.isdigit():
			print("Version must be a numeric value (e.g., 1, 2, 3...).")
			version = None  # type: ignore[assignment]
	
	return workspace, project, version  # type: ignore[return-value]


def try_upload_with_sdk(version_obj, weights_path: Path) -> bool:
	"""Attempt to upload using possible SDK methods. Return True on success."""
	candidates = [
		"upload_model",
		"upload_checkpoint",
		"upload_weights",
		"upload_model_weights",
	]
	for name in candidates:
		method = getattr(version_obj, name, None)
		if callable(method):
			try:
				print(f"Attempting upload via SDK method: {name}(...)")
				result = method(str(weights_path))  # type: ignore[misc]
				# Best-effort success message
				print("Upload response:", result)
				return True
			except Exception as e:  # noqa: BLE001
				print(f"SDK method {name} failed: {e}")
	return False


def main() -> None:
	rf = get_roboflow_client()

	# Pick a weights file from models directory
	models_dir = Path(__file__).resolve().parents[1] / "models"
	files = list_weight_files(models_dir)
	if not files:
		print(
			f"No supported weights found in {models_dir}.\n"
			f"Supported extensions: {', '.join(SUPPORTED_WEIGHT_EXTS)}"
		)
		return

	weights_path = prompt_select_file(files)
	if not weights_path:
		print("No file selected. Exiting.")
		return

	workspace_id, project_id, version_num = resolve_project_ids(rf)
	model_type = prompt_model_type()

	print(
		f"\nUsing workspace={workspace_id}, project={project_id}, version={version_num}"
	)
	print(f"Model type: {model_type}")
	print(f"Uploading: {weights_path.name}")
	print()

	# Navigate to project/version
	ws = rf.workspace(workspace_id)
	proj = ws.project(project_id)
	ver = proj.version(int(version_num))

	# Try SDK-based upload methods
	uploaded = try_upload_with_sdk(ver, weights_path)
	if uploaded:
		print("Weights upload completed via SDK.")
		return

	# Fallback if SDK upload method not available
	print(
		"Could not find a supported upload method in the Roboflow SDK for this object.\n"
		"If your Roboflow plan supports model uploads, please update the SDK to the latest version,\n"
		"or upload via the Roboflow UI/CLI for your project/version."
	)


if __name__ == "__main__":
	main()

