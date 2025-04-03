"""Caching utilities for BigQuery Cost Intelligence Engine.

This module provides caching functionality to improve API response times and reduce
repeated processing of expensive operations.
"""

import time
import hashlib
import json
import logging
import threading
import functools
from typing import Any, Dict, Optional, Callable, Tuple, Union, List
from datetime import datetime, timedelta

from .logging import setup_logger

logger = setup_logger(__name__)

# Global cache instance
_CACHE = {}
_CACHE_TTL = {}
_CACHE_HITS = {}
_CACHE_MISSES = {}
_CACHE_LOCK = threading.RLock()

# Default cache TTL in seconds
DEFAULT_TTL = 300  # 5 minutes


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    with _CACHE_LOCK:
        total_entries = len(_CACHE)
        total_hits = sum(_CACHE_HITS.values())
        total_misses = sum(_CACHE_MISSES.values())
        
        # Get sizes by category
        categories = {}
        for key in _CACHE:
            category = key.split(':', 1)[0] if ':' in key else 'unknown'
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        # Calculate hit rates by category
        hit_rates = {}
        for category in categories:
            category_hits = sum(hits for cache_key, hits in _CACHE_HITS.items() 
                               if cache_key.startswith(f"{category}:"))
            category_misses = sum(misses for cache_key, misses in _CACHE_MISSES.items() 
                                 if cache_key.startswith(f"{category}:"))
            total = category_hits + category_misses
            hit_rates[category] = (category_hits / total) if total > 0 else 0
        
        return {
            'total_entries': total_entries,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': (total_hits / (total_hits + total_misses)) if (total_hits + total_misses) > 0 else 0,
            'categories': categories,
            'category_hit_rates': hit_rates,
            'memory_usage_estimate_kb': get_cache_memory_usage() // 1024
        }


def get_cache_memory_usage() -> int:
    """Estimate the memory usage of the cache.
    
    Returns:
        Estimated memory usage in bytes
    """
    import sys
    
    with _CACHE_LOCK:
        # Estimate size of cache keys
        key_size = sum(sys.getsizeof(key) for key in _CACHE)
        
        # Estimate size of cache values (rough estimate)
        value_size = 0
        for value in _CACHE.values():
            try:
                # For JSON-serializable values, get a good estimate
                value_size += len(json.dumps(value).encode('utf-8'))
            except (TypeError, OverflowError):
                # For non-serializable values, use sys.getsizeof as fallback
                value_size += sys.getsizeof(value)
        
        # Add size of TTL and hit/miss dictionaries
        ttl_size = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in _CACHE_TTL.items())
        hits_size = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in _CACHE_HITS.items())
        misses_size = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in _CACHE_MISSES.items())
        
        return key_size + value_size + ttl_size + hits_size + misses_size


def clear_cache(category: Optional[str] = None) -> int:
    """Clear cache entries, optionally by category.
    
    Args:
        category: Optional category prefix to clear
        
    Returns:
        Number of cache entries cleared
    """
    with _CACHE_LOCK:
        if category:
            # Clear entries for specific category
            keys_to_remove = [key for key in _CACHE if key.startswith(f"{category}:")]
            for key in keys_to_remove:
                del _CACHE[key]
                if key in _CACHE_TTL:
                    del _CACHE_TTL[key]
            
            logger.info(f"Cleared {len(keys_to_remove)} cache entries for category '{category}'")
            return len(keys_to_remove)
        else:
            # Clear all cache
            count = len(_CACHE)
            _CACHE.clear()
            _CACHE_TTL.clear()
            logger.info(f"Cleared all {count} cache entries")
            return count


def clean_expired_cache(max_age_seconds: Optional[int] = None) -> int:
    """Clean expired cache entries.
    
    Args:
        max_age_seconds: Optional maximum age in seconds, defaults to DEFAULT_TTL
        
    Returns:
        Number of cache entries cleaned
    """
    current_time = time.time()
    max_age = max_age_seconds or DEFAULT_TTL
    
    with _CACHE_LOCK:
        # Find expired entries
        expired_keys = [
            key for key, ttl in _CACHE_TTL.items() 
            if ttl <= current_time or (current_time - ttl) > max_age
        ]
        
        # Remove expired entries
        for key in expired_keys:
            if key in _CACHE:
                del _CACHE[key]
            if key in _CACHE_TTL:
                del _CACHE_TTL[key]
        
        logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
        return len(expired_keys)


def cache_key(category: str, *args: Any, **kwargs: Any) -> str:
    """Generate a cache key from category and arguments.
    
    Args:
        category: Cache category
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key
        
    Returns:
        Cache key string
    """
    # Create a hash of the arguments
    key_parts = [category]
    
    # Add positional args
    for arg in args:
        if arg is not None:
            key_parts.append(str(arg))
    
    # Add keyword args (sorted for deterministic keys)
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if v is not None:
            key_parts.append(f"{k}={v}")
    
    # Join and hash
    key_str = ':'.join(key_parts)
    
    # For very long keys, create a hash instead
    if len(key_str) > 250:
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        key_str = f"{category}:hash:{key_hash}"
    
    return key_str


def cache_result(category: str, ttl_seconds: int = DEFAULT_TTL):
    """Decorator to cache function results.
    
    Args:
        category: Cache category
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip cache for certain keyword arguments
            if kwargs.get('skip_cache', False):
                if 'skip_cache' in kwargs:
                    del kwargs['skip_cache']
                return func(*args, **kwargs)
            
            # Generate cache key
            key = cache_key(f"{category}:{func.__name__}", *args, **kwargs)
            
            # Check cache
            with _CACHE_LOCK:
                if key in _CACHE and key in _CACHE_TTL:
                    # Check if cache is still valid
                    if time.time() < _CACHE_TTL[key]:
                        # Update hit counter
                        if key not in _CACHE_HITS:
                            _CACHE_HITS[key] = 0
                        _CACHE_HITS[key] += 1
                        
                        logger.debug(f"Cache hit for {key}")
                        return _CACHE[key]
                
                # Update miss counter
                if key not in _CACHE_MISSES:
                    _CACHE_MISSES[key] = 0
                _CACHE_MISSES[key] += 1
            
            # Execute function
            logger.debug(f"Cache miss for {key}, executing function")
            result = func(*args, **kwargs)
            
            # Store in cache
            with _CACHE_LOCK:
                _CACHE[key] = result
                _CACHE_TTL[key] = time.time() + ttl_seconds
            
            return result
        
        return wrapper
    
    return decorator


def get_from_cache(key: str) -> Tuple[bool, Any]:
    """Get a value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Tuple of (found, value)
    """
    with _CACHE_LOCK:
        if key in _CACHE and key in _CACHE_TTL:
            # Check if cache is still valid
            if time.time() < _CACHE_TTL[key]:
                # Update hit counter
                if key not in _CACHE_HITS:
                    _CACHE_HITS[key] = 0
                _CACHE_HITS[key] += 1
                
                logger.debug(f"Cache hit for {key}")
                return True, _CACHE[key]
        
        # Update miss counter
        if key not in _CACHE_MISSES:
            _CACHE_MISSES[key] = 0
        _CACHE_MISSES[key] += 1
        
        logger.debug(f"Cache miss for {key}")
        return False, None


def set_in_cache(key: str, value: Any, ttl_seconds: int = DEFAULT_TTL) -> None:
    """Store a value in cache.
    
    Args:
        key: Cache key
        value: Value to store
        ttl_seconds: Time-to-live in seconds
    """
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE_TTL[key] = time.time() + ttl_seconds
        logger.debug(f"Stored in cache: {key}")


# Start periodic cache cleaning
def _start_cache_cleaning():
    """Start periodic cache cleaning in a background thread."""
    def cleaning_task():
        while True:
            try:
                # Sleep first to avoid early cleaning
                time.sleep(300)  # 5 minutes
                
                # Clean expired entries
                cleaned = clean_expired_cache()
                
                # Log stats if significant cleaning occurred
                if cleaned > 10:
                    stats = get_cache_stats()
                    logger.info(f"Cache stats after cleaning: {stats['total_entries']} entries, "
                               f"{stats['memory_usage_estimate_kb']} KB")
            except Exception as e:
                logger.error(f"Error in cache cleaning task: {e}")
    
    # Start cleaning thread
    cleaning_thread = threading.Thread(target=cleaning_task, daemon=True)
    cleaning_thread.start()
    logger.info("Started cache cleaning background thread")

# Initialize cache cleaning on module import
_start_cache_cleaning()