"""
Configuration Manager for Robot PnP System

Handles YAML configuration loading, validation, hot-reload support, and path resolution.
Provides type-safe access to configuration parameters with sensible defaults.

Features:
- YAML file loading with validation
- Hot-reload capability for runtime config updates
- Path resolution (absolute/relative to repo root)
- Transform matrix generation from config
- Nested config access with defaults
- Change detection and notification

by: Aldrick T, 2025
for Team YEA
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import hashlib


# Repository root for path resolution
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pickafresa_robot.robot_system.transform_utils import TransformUtils


class ConfigManager:
    """
    Configuration manager with hot-reload support.
    
    Handles loading, validation, and runtime updates of YAML configuration files.
    """
    
    # Configs that support hot-reload (can be changed without restarting controller)
    HOT_RELOADABLE_KEYS = {
        'run_mode',
        'transforms.pick_offset',  # Pick/prepick/place offsets
        'post_pick',
        'mqtt',
        'vision_service',
        'multi_berry',
        'movement_speeds',
        'safety'
    }
    
    # Configs that require controller restart
    COLD_KEYS = {
        'robodk',
        'transforms',  # TCP transforms should not change during operation
        'speed_profiles'  # Profile definitions (but profile selection is hot-reloadable)
    }
    
    def __init__(self, config_path: Path, logger=None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
            logger: Optional logger instance
        """
        self.config_path = Path(config_path)
        self.logger = logger
        self.config: Dict[str, Any] = {}
        self._file_hash: Optional[str] = None
        self._change_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Load initial configuration
        self.load()
    
    def _log(self, level: str, message: str):
        """Internal logging helper."""
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def load(self) -> bool:
        """
        Load configuration from YAML file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.config_path.exists():
            self._log("error", f"Config file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            
            # Compute file hash for change detection
            self._file_hash = self._compute_file_hash()
            
            self._log("info", f"[OK] Loaded configuration from: {self.config_path.name}")
            return True
        
        except yaml.YAMLError as e:
            self._log("error", f"Failed to parse YAML: {e}")
            return False
        except Exception as e:
            self._log("error", f"Failed to load config: {e}")
            return False
    
    def reload(self) -> bool:
        """
        Reload configuration from file if changed.
        
        Returns:
            True if config was reloaded (file changed), False if unchanged or error
        """
        if not self.has_changed():
            self._log("debug", "Config file unchanged, skipping reload")
            return False
        
        self._log("info", "Config file changed, reloading...")
        
        old_config = self.config.copy()
        
        if self.load():
            # Notify callbacks of changes
            self._notify_changes(old_config, self.config)
            return True
        
        return False
    
    def has_changed(self) -> bool:
        """
        Check if config file has changed since last load.
        
        Returns:
            True if file has changed
        """
        current_hash = self._compute_file_hash()
        return current_hash != self._file_hash
    
    def _compute_file_hash(self) -> Optional[str]:
        """Compute MD5 hash of config file for change detection."""
        if not self.config_path.exists():
            return None
        
        try:
            with open(self.config_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def register_change_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback to be notified of config changes.
        
        Args:
            callback: Function to call with new config when changes detected
        """
        self._change_callbacks.append(callback)
    
    def _notify_changes(self, old_config: Dict, new_config: Dict):
        """Notify registered callbacks of config changes."""
        # Detect which keys changed
        changed_keys = set()
        
        for key in set(old_config.keys()) | set(new_config.keys()):
            if old_config.get(key) != new_config.get(key):
                changed_keys.add(key)
        
        if changed_keys:
            self._log("info", f"Config changes detected in: {', '.join(changed_keys)}")
            
            # Notify callbacks
            for callback in self._change_callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    self._log("error", f"Error in change callback: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dots, e.g., "robodk.robot_model")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value (in-memory only, does not save to file).
        
        Args:
            key: Configuration key (supports nested keys with dots)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def is_hot_reloadable(self, key: str) -> bool:
        """
        Check if a configuration key supports hot-reload.
        
        Args:
            key: Top-level config key
        
        Returns:
            True if key can be hot-reloaded
        """
        return key in self.HOT_RELOADABLE_KEYS
    
    def resolve_path(self, path_str: str) -> Path:
        """
        Resolve path (absolute or relative to repo root).
        
        Args:
            path_str: Path string from config
        
        Returns:
            Absolute Path object
        """
        path = Path(path_str)
        
        if path.is_absolute():
            return path
        else:
            return (REPO_ROOT / path).resolve()
    
    def get_transform_matrix(self, transform_key: str):
        """
        Get transformation matrix from config.
        
        Args:
            transform_key: Key under 'transforms' section (e.g., "camera_tcp", "gripper_tcp")
        
        Returns:
            4x4 numpy transformation matrix (in meters)
        """
        transform_config = self.get(f'transforms.{transform_key}', {})
        
        translation_mm = transform_config.get('translation_mm', [0, 0, 0])
        rotation_deg = transform_config.get('rotation_deg', [0, 0, 0])
        
        return TransformUtils.create_transform_matrix(
            translation=translation_mm,
            rotation_deg=rotation_deg,
            input_units="mm"
        )
    
    def get_offset_config(self, offset_key: str) -> Dict[str, Any]:
        """
        Get offset configuration (translation + rotation).
        
        Args:
            offset_key: Nested key path (e.g., "transforms.pick_offset.prepick")
        
        Returns:
            Dictionary with offset_mm and rotation_deg
        """
        # Parse the nested key
        parts = offset_key.split('.')
        
        # Try to extract the offset name from the last part
        if len(parts) >= 2:
            # Get the parent config section
            parent_path = '.'.join(parts[:-1])
            offset_name = parts[-1]
            
            parent_config = self.get(parent_path, {})
            
            offset_mm = parent_config.get(f'{offset_name}_offset_mm', [0, 0, 0])
            rotation_deg = parent_config.get(f'{offset_name}_rotation_deg', None)
            
            return {
                'offset_mm': offset_mm,
                'rotation_deg': rotation_deg
            }
        else:
            # Fallback: just get the key directly
            value = self.get(offset_key, [0, 0, 0])
            return {
                'offset_mm': value if isinstance(value, list) else [0, 0, 0],
                'rotation_deg': None
            }
    
    def validate(self) -> bool:
        """
        Validate configuration for required keys and value ranges.
        
        Returns:
            True if valid, False otherwise (logs errors)
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ['robodk', 'transforms']
        for key in required_keys:
            if key not in self.config:
                errors.append(f"Missing required key: {key}")
        
        # Validate run_mode
        run_mode = self.get('run_mode', 'manual_confirm')
        if run_mode not in ['manual_confirm', 'autonomous']:
            errors.append(f"Invalid run_mode: {run_mode}")
        
        # Validate RoboDK station file exists
        station_file = self.get('robodk.station_file')
        if station_file:
            station_path = self.resolve_path(station_file)
            if not station_path.exists():
                errors.append(f"RoboDK station file not found: {station_path}")
        
        # Validate TCP transforms have required fields
        for tcp_key in ['camera_tcp', 'gripper_tcp']:
            tcp_config = self.get(f'transforms.{tcp_key}', {})
            if 'translation_mm' not in tcp_config:
                errors.append(f"transforms.{tcp_key} missing translation_mm")
            if 'rotation_deg' not in tcp_config:
                errors.append(f"transforms.{tcp_key} missing rotation_deg")
        
        # Log errors
        if errors:
            self._log("error", "Configuration validation failed:")
            for error in errors:
                self._log("error", f"  - {error}")
            return False
        
        self._log("info", "[OK] Configuration validation passed")
        return True
    
    def save(self, path: Optional[Path] = None) -> bool:
        """
        Save current configuration to YAML file.
        
        Args:
            path: Optional path to save to (defaults to original config_path)
        
        Returns:
            True if saved successfully
        """
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            # Update hash if saving to original path
            if save_path == self.config_path:
                self._file_hash = self._compute_file_hash()
            
            self._log("info", f"[OK] Saved configuration to: {save_path}")
            return True
        
        except Exception as e:
            self._log("error", f"Failed to save config: {e}")
            return False
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of current configuration.
        
        Returns:
            Multi-line string summary
        """
        lines = [
            "=" * 70,
            "CONFIGURATION SUMMARY",
            "=" * 70,
            f"Config file: {self.config_path.name}",
            f"Last modified: {datetime.fromtimestamp(self.config_path.stat().st_mtime)}",
            "",
            f"Run mode: {self.get('run_mode', 'N/A')}",
            f"Simulation: {self.get('robodk.simulation_mode', 'N/A')}",
            f"Robot model: {self.get('robodk.robot_model', 'N/A')}",
            "",
            "MQTT Gripper:",
            f"  Enabled: {self.get('mqtt.enabled', False)}",
            f"  Broker: {self.get('mqtt.broker_address', 'N/A')}:{self.get('mqtt.broker_port', 'N/A')}",
            "",
            "Vision Service:",
            f"  Host: {self.get('vision_service.host', '127.0.0.1')}",
            f"  Port: {self.get('vision_service.port', 5555)}",
            "",
            "Multi-Berry Mode:",
            f"  Mode: {self.get('multi_berry.mode', 'N/A')}",
            f"  Max berries: {self.get('multi_berry.max_berries', 'N/A')}",
            "=" * 70
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ConfigManager(path={self.config_path}, keys={len(self.config)})"


# Example usage and tests
if __name__ == "__main__":
    print("Config Manager - Unit Tests")
    print("=" * 70)
    
    # Test 1: Load config
    print("\n[Test 1] Load configuration")
    config_path = REPO_ROOT / "pickafresa_robot/configs/robot_pnp_config.yaml"
    
    if not config_path.exists():
        print(f"[SKIP] Config file not found: {config_path}")
    else:
        manager = ConfigManager(config_path)
        
        print(f"Loaded: {manager}")
        print(f"Run mode: {manager.get('run_mode')}")
        print(f"Robot model: {manager.get('robodk.robot_model')}")
        
        # Test 2: Path resolution
        print("\n[Test 2] Path resolution")
        station_file = manager.get('robodk.station_file')
        resolved_path = manager.resolve_path(station_file)
        print(f"Station file (raw): {station_file}")
        print(f"Station file (resolved): {resolved_path}")
        print(f"Exists: {resolved_path.exists()}")
        
        # Test 3: Transform matrices
        print("\n[Test 3] Transform matrices")
        T_flange_camera = manager.get_transform_matrix('camera_tcp')
        print(f"T_flange_camera:\n{T_flange_camera}")
        
        T_flange_gripper = manager.get_transform_matrix('gripper_tcp')
        print(f"T_flange_gripper:\n{T_flange_gripper}")
        
        # Test 4: Offset config
        print("\n[Test 4] Offset configuration")
        prepick_offset = manager.get_offset_config('transforms.pick_offset.prepick')
        print(f"Prepick offset: {prepick_offset}")
        
        pick_offset = manager.get_offset_config('transforms.pick_offset.pick')
        print(f"Pick offset: {pick_offset}")
        
        place_offset = manager.get_offset_config('transforms.pick_offset.place')
        print(f"Place offset: {place_offset}")
        
        # Test 5: Validation
        print("\n[Test 5] Configuration validation")
        is_valid = manager.validate()
        print(f"Valid: {is_valid}")
        
        # Test 6: Summary
        print("\n[Test 6] Configuration summary")
        print(manager.get_summary())
        
        # Test 7: Hot-reload check
        print("\n[Test 7] Hot-reload capability")
        test_keys = ['run_mode', 'robodk', 'transforms.pick_offset', 'transforms', 'post_pick']
        for key in test_keys:
            hot = manager.is_hot_reloadable(key)
            print(f"  {key}: {'[OK] HOT' if hot else '[FAIL] COLD (requires restart)'}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
