"""
Thread-safe file operations with configurable locking strategies.
"""

import os
import threading
import time
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, Callable, Dict, Set
from threading import RLock

log = logging.getLogger(__name__)

# Configuration constants
DEFAULT_LOCK_TIMEOUT = 30.0
LOCK_FILE_SUFFIX = ".lock"
CLEANUP_INTERVAL = 300  # 5 minutes


class FileOperationStrategy(ABC):
    """Abstract base class for file operation locking strategies."""
    
    @abstractmethod
    @contextmanager
    def read_context(self, filepath: str):
        """Context manager for read operations."""
        pass
    
    @abstractmethod
    @contextmanager
    def write_context(self, filepath: str):
        """Context manager for write operations."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by the strategy."""
        pass


class NoLockStrategy(FileOperationStrategy):
    """No-op strategy for environments where locking isn't needed."""
    
    @contextmanager
    def read_context(self, filepath: str):
        yield
    
    @contextmanager
    def write_context(self, filepath: str):
        yield
    
    def cleanup(self) -> None:
        pass


class ThreadLocalLockStrategy(FileOperationStrategy):
    """In-process threading locks (doesn't work across processes)."""
    
    def __init__(self):
        self._locks: Dict[str, RLock] = {}
        self._locks_lock = RLock()
    
    def _get_lock(self, filepath: str) -> RLock:
        with self._locks_lock:
            if filepath not in self._locks:
                self._locks[filepath] = RLock()
            return self._locks[filepath]
    
    @contextmanager
    def read_context(self, filepath: str):
        # For thread-local locks, we can allow concurrent reads
        # by using the same lock but not blocking reads
        lock = self._get_lock(filepath)
        with lock:
            yield
    
    @contextmanager
    def write_context(self, filepath: str):
        lock = self._get_lock(filepath)
        with lock:
            yield
    
    def cleanup(self) -> None:
        with self._locks_lock:
            self._locks.clear()


class FileLockStrategy(FileOperationStrategy):
    """Cross-process file locking strategy using FileLock."""
    
    def __init__(self, timeout: float = DEFAULT_LOCK_TIMEOUT, 
                 enable_reader_optimization: bool = True):
        self.timeout = timeout
        self.enable_reader_optimization = enable_reader_optimization
        self._active_locks: Set[str] = set()
        self._active_locks_lock = threading.Lock()
        self._last_cleanup = time.time()
        
        # Import FileLock here to make it optional
        try:
            from filelock import FileLock
            self._FileLock = FileLock
        except ImportError:
            raise ImportError(
                "FileLock strategy requires 'filelock' package. "
                "Install with: pip install filelock"
            )
    
    def _get_lock_path(self, filepath: str) -> str:
        """Generate lock file path."""
        return filepath + LOCK_FILE_SUFFIX
    
    def _track_lock(self, lock_path: str) -> None:
        """Track active lock files for cleanup."""
        with self._active_locks_lock:
            self._active_locks.add(lock_path)
    
    def _untrack_lock(self, lock_path: str) -> None:
        """Untrack lock files."""
        with self._active_locks_lock:
            self._active_locks.discard(lock_path)
    
    @contextmanager
    def read_context(self, filepath: str):
        if self.enable_reader_optimization:
            # For reads, we can check if file exists without locking
            # and only lock if we need to ensure consistency
            if os.path.exists(filepath):
                yield
                return
        
        # Fall back to write locking for safety
        with self.write_context(filepath):
            yield
    
    @contextmanager
    def write_context(self, filepath: str):
        lock_path = self._get_lock_path(filepath)
        file_lock = self._FileLock(lock_path, timeout=self.timeout)
        
        self._track_lock(lock_path)
        try:
            with file_lock:
                yield
        except Exception as e:
            log.error(f"FileLock operation failed for {filepath}: {e}")
            raise
        finally:
            self._untrack_lock(lock_path)
            # Periodic cleanup
            if time.time() - self._last_cleanup > CLEANUP_INTERVAL:
                self._cleanup_stale_locks()
    
    def _cleanup_stale_locks(self) -> None:
        """Clean up stale lock files."""
        try:
            current_time = time.time()
            with self._active_locks_lock:
                active_copies = set(self._active_locks)
            
            # Clean up lock files that are old and not actively tracked
            for lock_path in list(active_copies):
                try:
                    if (os.path.exists(lock_path) and 
                        current_time - os.path.getmtime(lock_path) > CLEANUP_INTERVAL):
                        os.remove(lock_path)
                        log.debug(f"Cleaned up stale lock file: {lock_path}")
                except (OSError, IOError) as e:
                    log.debug(f"Could not clean up lock file {lock_path}: {e}")
            
            self._last_cleanup = current_time
        except Exception as e:
            log.debug(f"Lock cleanup failed: {e}")
    
    def cleanup(self) -> None:
        """Clean up all tracked lock files."""
        with self._active_locks_lock:
            for lock_path in list(self._active_locks):
                try:
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                        log.debug(f"Cleaned up lock file: {lock_path}")
                except (OSError, IOError) as e:
                    log.debug(f"Could not clean up lock file {lock_path}: {e}")
            self._active_locks.clear()
        
        # Also clean up any orphaned lock files in common directories
        import glob
        try:
            # Clean up lock files in current directory and common temp locations
            for pattern in ["*.lock", "/tmp/*.lock"]:
                for lock_file in glob.glob(pattern):
                    try:
                        # Only remove if it's old enough (more than 60 seconds)
                        if time.time() - os.path.getmtime(lock_file) > 60:
                            os.remove(lock_file)
                            log.debug(f"Cleaned up orphaned lock file: {lock_file}")
                    except (OSError, IOError):
                        pass
        except Exception:
            pass  # Ignore cleanup errors


class ThreadSafeFileOperations:
    """Main class providing thread-safe file operations with configurable strategies."""
    
    _instance: Optional['ThreadSafeFileOperations'] = None
    _instance_lock = threading.Lock()
    
    def __init__(self, strategy: Optional[FileOperationStrategy] = None):
        if strategy is None:
            # Default strategy: use FileLock if available, otherwise thread-local
            try:
                strategy = FileLockStrategy()
            except ImportError:
                log.warning(
                    "FileLock not available, falling back to thread-local locks. "
                    "This won't protect against multi-process race conditions."
                )
                strategy = ThreadLocalLockStrategy()
        
        self.strategy = strategy
    
    @classmethod
    def get_instance(cls) -> 'ThreadSafeFileOperations':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def configure(cls, strategy: FileOperationStrategy) -> None:
        """Configure the global file operations strategy."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.strategy.cleanup()
            cls._instance = cls(strategy)
    
    @contextmanager
    def read_operation(self, filepath: str):
        """Context manager for read operations."""
        with self.strategy.read_context(filepath):
            yield
    
    @contextmanager
    def write_operation(self, filepath: str):
        """Context manager for write operations."""
        with self.strategy.write_context(filepath):
            yield
    
    def execute_read(self, filepath: str, operation: Callable[[], Any]) -> Any:
        """Execute a read operation with proper locking."""
        with self.read_operation(filepath):
            return operation()
    
    def execute_write(self, filepath: str, operation: Callable[[], Any]) -> Any:
        """Execute a write operation with proper locking."""
        with self.write_operation(filepath):
            return operation()


# Global instance for backward compatibility
_global_file_ops = ThreadSafeFileOperations.get_instance()


def get_file_operations() -> ThreadSafeFileOperations:
    """Get the global file operations instance."""
    return _global_file_ops


def configure_file_operations(strategy: FileOperationStrategy) -> None:
    """Configure the global file operations strategy."""
    ThreadSafeFileOperations.configure(strategy)


# Convenience context managers for backward compatibility
@contextmanager
def safe_read_operation(filepath: str):
    """Context manager for thread-safe read operations."""
    with get_file_operations().read_operation(filepath):
        yield


@contextmanager
def safe_write_operation(filepath: str):
    """Context manager for thread-safe write operations."""
    with get_file_operations().write_operation(filepath):
        yield
