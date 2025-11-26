"""
ROS2-Style Logger for Robot PnP Testing Tool

Provides logging functionality mimicking ROS2 format:
[timestamp] [level] [node_name]: message

Features:
- Console and file logging
- Configurable log levels
- Thread-safe operation
- Microsecond timestamp precision

by: Aldrick T, 2025
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ROS2StyleLogger:
    """Logger with ROS2-style formatting."""
    
    # ANSI color codes for console output
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARN': '\033[33m',     # Yellow
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(
        self,
        node_name: str = "robot_pnp",
        log_file: Optional[Path] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f",
        overwrite_log: bool = True,
        mqtt_callback: Optional[callable] = None
    ):
        """
        Initialize ROS2-style logger.
        
        Args:
            node_name: Name of the node/module (appears in log messages)
            log_file: Path to log file (None = no file logging)
            console_level: Logging level for console output
            file_level: Logging level for file output
            timestamp_format: Format string for timestamps
            overwrite_log: If True, overwrite log file on start; if False, append
            mqtt_callback: Optional callback function(level, message) to forward logs to MQTT
        """
        self.node_name = node_name
        self.timestamp_format = timestamp_format
        self.mqtt_callback = mqtt_callback
        
        # Create logger
        self.logger = logging.getLogger(node_name)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels
        self.logger.handlers.clear()  # Remove existing handlers
        
        # Console handler with ROS2 formatting
        if console_level:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, console_level.upper()))
            console_handler.setFormatter(self._create_console_formatter())
            self.logger.addHandler(console_handler)
        
        # File handler with ROS2 formatting (no colors)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_mode = 'w' if overwrite_log else 'a'  # 'w' = overwrite, 'a' = append
            file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_handler.setFormatter(self._create_file_formatter())
            self.logger.addHandler(file_handler)
    
    def _create_console_formatter(self) -> logging.Formatter:
        """Create colored formatter for console output."""
        class ColoredFormatter(logging.Formatter):
            def __init__(self, node_name: str, timestamp_format: str, colors: dict):
                super().__init__()
                self.node_name = node_name
                self.timestamp_format = timestamp_format
                self.colors = colors
            
            def format(self, record: logging.LogRecord) -> str:
                timestamp = datetime.now().strftime(self.timestamp_format)[:-3]  # Truncate to ms
                level = record.levelname
                color = self.colors.get(level, self.colors['RESET'])
                reset = self.colors['RESET']
                
                # ROS2 format: [timestamp] [level] [node]: message
                return f"{color}[{timestamp}] [{level}] [{self.node_name}]:{reset} {record.getMessage()}"
        
        return ColoredFormatter(self.node_name, self.timestamp_format, self.COLORS)
    
    def _create_file_formatter(self) -> logging.Formatter:
        """Create plain formatter for file output (no colors)."""
        class PlainFormatter(logging.Formatter):
            def __init__(self, node_name: str, timestamp_format: str):
                super().__init__()
                self.node_name = node_name
                self.timestamp_format = timestamp_format
            
            def format(self, record: logging.LogRecord) -> str:
                timestamp = datetime.now().strftime(self.timestamp_format)[:-3]
                level = record.levelname
                return f"[{timestamp}] [{level}] [{self.node_name}]: {record.getMessage()}"
        
        return PlainFormatter(self.node_name, self.timestamp_format)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
        if self.mqtt_callback:
            self.mqtt_callback("DEBUG", message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
        if self.mqtt_callback:
            self.mqtt_callback("INFO", message)
    
    def warn(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
        if self.mqtt_callback:
            self.mqtt_callback("WARN", message)
    
    def warning(self, message: str) -> None:
        """Log warning message (alias)."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
        if self.mqtt_callback:
            self.mqtt_callback("ERROR", message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)
        if self.mqtt_callback:
            self.mqtt_callback("CRITICAL", message)
    
    def set_level(self, level: str) -> None:
        """Change logging level dynamically."""
        self.logger.setLevel(getattr(logging, level.upper()))


def create_logger(
    node_name: str = "robot_pnp",
    log_dir: Optional[Path] = None,
    log_prefix: str = "robot_pnp",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    use_timestamp: bool = False,
    overwrite_log: bool = True,
    mqtt_callback: Optional[callable] = None
) -> ROS2StyleLogger:
    """
    Factory function to create a ROS2-style logger.
    
    Args:
        node_name: Name of the node
        log_dir: Directory for log files (None = no file logging)
        log_prefix: Prefix for log filename
        console_level: Console logging level
        file_level: File logging level
        use_timestamp: If True, add timestamp to filename; if False, use fixed name
        overwrite_log: If True, overwrite log on start; if False, append
        mqtt_callback: Optional callback function(level, message) to forward logs to MQTT
    
    Returns:
        Configured ROS2StyleLogger instance
    """
    log_file = None
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{log_prefix}_{timestamp}.log"
        else:
            # Fixed log filename (will be overwritten on each run if overwrite_log=True)
            log_file = log_dir / f"{log_prefix}.log"
    
    return ROS2StyleLogger(
        node_name=node_name,
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
        overwrite_log=overwrite_log,
        mqtt_callback=mqtt_callback
    )


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = create_logger(
        node_name="test_node",
        log_dir=Path("logs"),
        console_level="DEBUG"
    )
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("Robot initialized successfully")
    logger.warn("Low battery warning")
    logger.error("Failed to connect to robot")
    logger.critical("Emergency stop activated!")
