"""
Persistent Statistics Manager for Robot PnP System

Tracks and persists operation statistics across service restarts.

by: Aldrick T, 2025
for Team YEA
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import threading


class PersistentStats:
    """
    Manages persistent statistics with automatic file I/O.
    Thread-safe for concurrent access.
    """
    
    def __init__(self, stats_file: Path):
        """
        Initialize persistent statistics.
        
        Args:
            stats_file: Path to JSON statistics file
        """
        self.stats_file = stats_file
        self.lock = threading.Lock()
        
        # Ensure directory exists
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing stats or initialize
        self.stats = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load statistics from file."""
        if not self.stats_file.exists():
            return self._create_default_stats()
        
        try:
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            
            # Ensure all required fields exist (for backward compatibility)
            default = self._create_default_stats()
            for key in default:
                if key not in stats:
                    stats[key] = default[key]
            
            return stats
        
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load stats file: {e}. Creating new stats.")
            return self._create_default_stats()
    
    def _create_default_stats(self) -> Dict[str, Any]:
        """Create default statistics structure."""
        return {
            'lifetime': {
                'requests_total': 0,
                'requests_success': 0,
                'requests_failed': 0,
                'picks_total': 0,
                'picks_success': 0,
                'picks_failed': 0,
                'multi_berry_runs': 0,
                'service_starts': 0,
                'first_start': None,
                'last_start': None
            },
            'current_session': {
                'requests_total': 0,
                'requests_success': 0,
                'requests_failed': 0,
                'picks_total': 0,
                'picks_success': 0,
                'picks_failed': 0,
                'multi_berry_runs': 0,
                'session_start': None
            }
        }
    
    def _save(self):
        """Save statistics to file."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save stats file: {e}")
    
    def start_session(self):
        """Mark the start of a new session."""
        with self.lock:
            now = datetime.now().isoformat()
            
            # Update lifetime stats
            self.stats['lifetime']['service_starts'] += 1
            if self.stats['lifetime']['first_start'] is None:
                self.stats['lifetime']['first_start'] = now
            self.stats['lifetime']['last_start'] = now
            
            # Reset current session
            self.stats['current_session'] = {
                'requests_total': 0,
                'requests_success': 0,
                'requests_failed': 0,
                'picks_total': 0,
                'picks_success': 0,
                'picks_failed': 0,
                'multi_berry_runs': 0,
                'session_start': now
            }
            
            self._save()
    
    def increment(self, category: str, field: str, amount: int = 1):
        """
        Increment a statistic field.
        
        Args:
            category: 'lifetime' or 'current_session'
            field: Field name (e.g., 'requests_total')
            amount: Amount to increment (default 1)
        """
        with self.lock:
            if category in self.stats and field in self.stats[category]:
                self.stats[category][field] += amount
                self._save()
    
    def increment_both(self, field: str, amount: int = 1):
        """
        Increment both lifetime and current session.
        
        Args:
            field: Field name to increment
            amount: Amount to increment (default 1)
        
        Note:
            If field doesn't exist, it will be auto-created with initial value of 0.
            This ensures backward compatibility when new stats fields are added.
        """
        with self.lock:
            # Auto-create missing fields (backward compatibility)
            if field not in self.stats['lifetime']:
                print(f"Warning: Stats field '{field}' not found. Auto-creating with value 0.")
                self.stats['lifetime'][field] = 0
            if field not in self.stats['current_session']:
                self.stats['current_session'][field] = 0
            
            self.stats['lifetime'][field] += amount
            self.stats['current_session'][field] += amount
            self._save()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self.lock:
            return self.stats.copy()
    
    def get_lifetime(self) -> Dict[str, Any]:
        """Get lifetime statistics."""
        with self.lock:
            return self.stats['lifetime'].copy()
    
    def get_session(self) -> Dict[str, Any]:
        """Get current session statistics."""
        with self.lock:
            return self.stats['current_session'].copy()


if __name__ == "__main__":
    # Test
    from pathlib import Path
    
    test_file = Path("/tmp/test_stats.json")
    stats = PersistentStats(test_file)
    
    print("Initial stats:", stats.get_stats())
    
    stats.start_session()
    stats.increment_both('requests_total')
    stats.increment_both('requests_success')
    stats.increment_both('picks_total')
    stats.increment_both('picks_success')
    
    print("\nAfter operations:", stats.get_stats())
    
    # Simulate restart
    stats2 = PersistentStats(test_file)
    print("\nAfter reload:", stats2.get_stats())
