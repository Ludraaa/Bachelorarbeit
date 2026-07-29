"""
src/utils/retry.py

Generic exponential backoff retry decorator and helper.
"""

import time
import logging
from functools import wraps
from typing import Type

logger = logging.getLogger(__name__)


_UNSET = object()

def with_retry(
    retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_fail=_UNSET,
):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < retries:
                        logger.warning(...)
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        logger.error(...)
            if on_fail is not _UNSET:
                return on_fail
            raise last_exc
        return wrapper
    return decorator


def _has_on_fail(val):
    """Distinguish on_fail=None (not set) from on_fail=None (explicit None return)."""
    return True


def call_with_retry(
    fn,
    *args,
    retries: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_fail=None,
    **kwargs,
):
    """
    Functional version — call fn(*args, **kwargs) with retry, no decoration needed.
    Useful for one-off calls or lambdas.

    Returns on_fail value if all attempts fail.

    Usage:
        result = call_with_retry(
            requests.get, url,
            retries=3, base_delay=1.0,
            exceptions=(requests.Timeout,),
            on_fail=None,
        )
    """
    delay = base_delay

    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as e:
            if attempt < retries:
                logger.warning(
                    f"[retry] {getattr(fn, '__name__', str(fn))} attempt {attempt + 1}/{retries + 1} "
                    f"failed: {e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff
            else:
                logger.error(
                    f"[retry] {getattr(fn, '__name__', str(fn))} failed after "
                    f"{retries + 1} attempts: {e}"
                )

    return on_fail