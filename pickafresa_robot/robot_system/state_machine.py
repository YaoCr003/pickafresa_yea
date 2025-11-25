"""
State Machine for Robot PnP Controller

Manages the operational states of the pick-and-place controller with thread-safe transitions.
Provides state validation, transition logging, and error handling.

States:
- IDLE: System ready, waiting for commands
- INITIALIZING: Starting up, connecting to subsystems
- MOVING: Robot in motion (non-picking movement)
- CAPTURING: Requesting vision data
- PICKING: Executing pick-and-place sequence
- ERROR: Recoverable error state
- EMERGENCY_STOP: Critical error, requires intervention
- SHUTDOWN: Graceful shutdown in progress

by: Aldrick T, 2025
for Team YEA
"""

import threading
from enum import Enum, auto
from typing import Optional, Callable, Dict, Any
from datetime import datetime


class RobotState(Enum):
    """Robot controller states."""
    
    IDLE = auto()              # Ready, waiting for commands
    INITIALIZING = auto()      # Starting up
    MOVING = auto()            # Robot moving (non-picking)
    CAPTURING = auto()         # Acquiring vision data
    PICKING = auto()           # Executing pick sequence
    ERROR = auto()             # Recoverable error
    EMERGENCY_STOP = auto()    # Critical error, safety stop
    SHUTDOWN = auto()          # Graceful shutdown


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class RobotStateMachine:
    """
    Thread-safe state machine for robot controller.
    
    Manages state transitions with validation and logging.
    """
    
    # Valid state transitions (from_state -> [allowed_to_states])
    VALID_TRANSITIONS: Dict[RobotState, list] = {
        RobotState.IDLE: [
            RobotState.MOVING,
            RobotState.CAPTURING,
            RobotState.PICKING,
            RobotState.SHUTDOWN,
            RobotState.ERROR
        ],
        RobotState.INITIALIZING: [
            RobotState.IDLE,
            RobotState.ERROR,
            RobotState.EMERGENCY_STOP
        ],
        RobotState.MOVING: [
            RobotState.IDLE,
            RobotState.CAPTURING,
            RobotState.PICKING,
            RobotState.ERROR,
            RobotState.EMERGENCY_STOP
        ],
        RobotState.CAPTURING: [
            RobotState.IDLE,
            RobotState.PICKING,
            RobotState.ERROR,
            RobotState.EMERGENCY_STOP
        ],
        RobotState.PICKING: [
            RobotState.IDLE,
            RobotState.MOVING,
            RobotState.ERROR,
            RobotState.EMERGENCY_STOP
        ],
        RobotState.ERROR: [
            RobotState.IDLE,
            RobotState.EMERGENCY_STOP,
            RobotState.SHUTDOWN
        ],
        RobotState.EMERGENCY_STOP: [
            RobotState.IDLE,  # Only after manual reset
            RobotState.SHUTDOWN
        ],
        RobotState.SHUTDOWN: []  # Terminal state
    }
    
    def __init__(self, initial_state: RobotState = RobotState.INITIALIZING, logger=None):
        """
        Initialize state machine.
        
        Args:
            initial_state: Starting state
            logger: Optional logger instance
        """
        self._state = initial_state
        self._lock = threading.Lock()
        self._logger = logger
        self._state_history: list = []
        self._callbacks: Dict[RobotState, list] = {}
        
        # Record initial state
        self._record_state(initial_state, "Initial state")
        
        self._log("info", f"State machine initialized in state: {initial_state.name}")
    
    def _log(self, level: str, message: str):
        """Internal logging helper."""
        if self._logger:
            getattr(self._logger, level)(message)
        # Always print critical messages
        if level in ["error", "warning"] or not self._logger:
            print(f"[{level.upper()}] {message}")
    
    @property
    def state(self) -> RobotState:
        """Get current state (thread-safe)."""
        with self._lock:
            return self._state
    
    @property
    def state_name(self) -> str:
        """Get current state name as string."""
        return self.state.name
    
    def is_state(self, state: RobotState) -> bool:
        """
        Check if currently in a specific state.
        
        Args:
            state: State to check
        
        Returns:
            True if in that state
        """
        return self.state == state
    
    def is_operational(self) -> bool:
        """Check if robot is in an operational state (can accept commands)."""
        return self.state in [RobotState.IDLE, RobotState.MOVING, RobotState.CAPTURING]
    
    def is_busy(self) -> bool:
        """Check if robot is busy (executing a task)."""
        return self.state in [RobotState.PICKING, RobotState.MOVING, RobotState.CAPTURING]
    
    def is_error_state(self) -> bool:
        """Check if in an error state."""
        return self.state in [RobotState.ERROR, RobotState.EMERGENCY_STOP]
    
    def can_transition_to(self, target_state: RobotState) -> bool:
        """
        Check if transition to target state is valid.
        
        Args:
            target_state: Desired state
        
        Returns:
            True if transition is allowed
        
        Note:
            This method assumes it's called within a lock context.
            Uses _state directly to avoid deadlock.
        """
        current = self._state  # Use _state directly to avoid deadlock when called within lock
        allowed = self.VALID_TRANSITIONS.get(current, [])
        return target_state in allowed
    
    def transition_to(self, target_state: RobotState, reason: str = "") -> bool:
        """
        Transition to a new state with validation.
        
        Args:
            target_state: Desired state
            reason: Optional reason for transition
        
        Returns:
            True if transition succeeded
        
        Raises:
            StateTransitionError: If transition is invalid
        """
        with self._lock:
            current = self._state
            
            # Check if transition is valid
            if not self.can_transition_to(target_state):
                error_msg = (
                    f"Invalid transition: {current.name} -> {target_state.name}"
                )
                self._log("error", error_msg)
                raise StateTransitionError(error_msg)
            
            # Perform transition
            self._state = target_state
            self._record_state(target_state, reason)
            
            # Log transition
            transition_msg = f"State: {current.name} -> {target_state.name}"
            if reason:
                transition_msg += f" (reason: {reason})"
            self._log("info", transition_msg)
            
            # Notify callbacks
            self._notify_callbacks(target_state)
            
            return True
    
    def force_transition(self, target_state: RobotState, reason: str = "Forced"):
        """
        Force transition without validation (use with caution!).
        
        Args:
            target_state: Desired state
            reason: Reason for forced transition
        """
        with self._lock:
            current = self._state
            self._state = target_state
            self._record_state(target_state, f"FORCED: {reason}")
            
            self._log("warning", f"FORCED transition: {current.name} -> {target_state.name} ({reason})")
            self._notify_callbacks(target_state)
    
    def to_idle(self, reason: str = ""):
        """Convenience method to transition to IDLE state."""
        self.transition_to(RobotState.IDLE, reason)
    
    def to_error(self, reason: str = ""):
        """Convenience method to transition to ERROR state."""
        self.transition_to(RobotState.ERROR, reason)
    
    def to_emergency_stop(self, reason: str = ""):
        """Convenience method to transition to EMERGENCY_STOP state."""
        self.transition_to(RobotState.EMERGENCY_STOP, reason)
    
    def reset_from_error(self, reason: str = "Manual reset"):
        """
        Reset from ERROR state to IDLE.
        
        Args:
            reason: Reason for reset
        
        Returns:
            True if reset succeeded
        """
        if not self.is_error_state():
            self._log("warning", f"Reset called but not in error state (current: {self.state_name})")
            return False
        
        try:
            self.transition_to(RobotState.IDLE, reason)
            return True
        except StateTransitionError:
            return False
    
    def register_callback(self, state: RobotState, callback: Callable[[RobotState], None]):
        """
        Register a callback for state entry.
        
        Args:
            state: State to monitor
            callback: Function to call when entering this state
        """
        if state not in self._callbacks:
            self._callbacks[state] = []
        self._callbacks[state].append(callback)
    
    def _notify_callbacks(self, state: RobotState):
        """Notify registered callbacks for state entry."""
        callbacks = self._callbacks.get(state, [])
        for callback in callbacks:
            try:
                callback(state)
            except Exception as e:
                self._log("error", f"Error in state callback: {e}")
    
    def _record_state(self, state: RobotState, reason: str):
        """Record state change in history."""
        entry = {
            "timestamp": datetime.now(),
            "state": state,
            "reason": reason
        }
        self._state_history.append(entry)
        
        # Keep only last 100 entries
        if len(self._state_history) > 100:
            self._state_history.pop(0)
    
    def get_history(self, limit: int = 10) -> list:
        """
        Get recent state history.
        
        Args:
            limit: Number of recent entries to return
        
        Returns:
            List of state history entries
        """
        return self._state_history[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status summary.
        
        Returns:
            Dictionary with state information
        """
        return {
            "state": self.state_name,
            "is_operational": self.is_operational(),
            "is_busy": self.is_busy(),
            "is_error": self.is_error_state(),
            "history_length": len(self._state_history)
        }
    
    def __repr__(self) -> str:
        return f"RobotStateMachine(state={self.state_name})"


# Example usage and tests
if __name__ == "__main__":
    print("Robot State Machine - Unit Tests")
    print("=" * 70)
    
    # Test 1: Initialization
    print("\n[Test 1] Initialization")
    sm = RobotStateMachine(initial_state=RobotState.INITIALIZING)
    print(f"Initial state: {sm.state_name}")
    print(f"Status: {sm.get_status()}")
    
    # Test 2: Valid transitions
    print("\n[Test 2] Valid transitions")
    sm.transition_to(RobotState.IDLE, "Initialization complete")
    print(f"After init: {sm.state_name}")
    
    sm.transition_to(RobotState.CAPTURING, "Requesting vision data")
    print(f"After capture request: {sm.state_name}")
    
    sm.transition_to(RobotState.PICKING, "Starting pick sequence")
    print(f"After pick start: {sm.state_name}")
    
    sm.transition_to(RobotState.IDLE, "Pick complete")
    print(f"After pick complete: {sm.state_name}")
    
    # Test 3: Invalid transition
    print("\n[Test 3] Invalid transition (should fail)")
    try:
        sm.transition_to(RobotState.SHUTDOWN, "")
        sm.transition_to(RobotState.PICKING, "Should not work from SHUTDOWN")
        print("[ERROR] Invalid transition was allowed!")
    except StateTransitionError as e:
        print(f"[OK] Transition blocked: {e}")
    
    # Test 4: Error handling
    print("\n[Test 4] Error handling")
    sm.to_error("Test error condition")
    print(f"Error state: {sm.state_name}, is_error={sm.is_error_state()}")
    
    sm.reset_from_error("Error resolved")
    print(f"After reset: {sm.state_name}, is_operational={sm.is_operational()}")
    
    # Test 5: Emergency stop
    print("\n[Test 5] Emergency stop")
    sm.transition_to(RobotState.PICKING, "Start pick")
    sm.to_emergency_stop("Critical failure")
    print(f"E-stop state: {sm.state_name}")
    
    # Test 6: Callbacks
    print("\n[Test 6] State callbacks")
    
    def on_idle_entry(state):
        print(f"  -> Callback triggered: Entered {state.name}")
    
    sm.register_callback(RobotState.IDLE, on_idle_entry)
    sm.reset_from_error("Reset from E-stop")
    
    # Test 7: History
    print("\n[Test 7] State history")
    history = sm.get_history(limit=5)
    print("Recent state changes:")
    for entry in history:
        timestamp = entry["timestamp"].strftime("%H:%M:%S")
        state = entry["state"].name
        reason = entry["reason"]
        print(f"  [{timestamp}] {state:<15} - {reason}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
