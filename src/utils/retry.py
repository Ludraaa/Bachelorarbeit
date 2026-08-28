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