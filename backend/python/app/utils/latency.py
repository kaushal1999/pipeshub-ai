"""
Latency measurement decorator for both sync and async functions.
"""

import functools
import inspect
import time
from typing import AsyncGenerator, Callable, Generator, Optional, TypeVar, Union

from app.utils.logger import create_logger

T = TypeVar('T')

_logger = create_logger("latency")

# Define constants for time thresholds
ONE_MILLISECOND = 0.001
ONE_SECOND = 1.0

def measure_latency(
    func: Optional[Callable] = None,
    *,
    logger: Optional[Callable] = None,
    log_level: str = "info",
    include_args: bool = False,
) -> Union[Callable[[Callable], Callable], Callable]:
    """
    Decorator to measure and optionally log the execution latency of functions.
    Works with synchronous functions, asynchronous functions, and async generators.

    Can be used with or without parentheses:
        @measure_latency
        def my_function():
            pass

        @measure_latency()
        def my_function():
            pass

        @measure_latency(logger=logger)
        def my_function():
            pass

    Args:
        func: The function to decorate (when used without parentheses).
        logger: Optional logger instance with methods like info(), debug(), etc.
                If None, latency is measured but not logged.
        log_level: Log level to use when logging (default: "info").
                   Should be one of: "debug", "info", "warning", "error".
        include_args: If True, includes function arguments in the log message (default: False).

    Returns:
        Decorated function that measures latency.

    Example:
        ```python
        # Basic usage (without parentheses)
        @measure_latency
        def sync_function(x, y):
            return x + y

        # Basic usage (with parentheses)
        @measure_latency()
        async def async_function(x, y):
            return x + y

        @measure_latency()
        async def async_generator_function(x, y):
            yield x
            yield y

        # With logging
        import logging
        logger = logging.getLogger(__name__)

        @measure_latency(logger=logger)
        def logged_function(x, y):
            return x + y

        # With custom log level
        @measure_latency(logger=logger, log_level="debug")
        def debug_logged_function(x, y):
            return x + y
        ```

    The decorator stores the latency in the function's return value as an attribute
    `_latency` (in seconds) if you need to access it programmatically.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        is_async = inspect.iscoroutinefunction(func)
        is_async_gen = inspect.isasyncgenfunction(func)
        is_sync_gen = inspect.isgeneratorfunction(func)

        if is_async_gen:
            # Handle async generator functions
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs) -> AsyncGenerator:
                start_time = time.perf_counter()
                try:
                    async for item in func(*args, **kwargs):
                        yield item
                finally:
                    elapsed_time = time.perf_counter() - start_time
                    _log_latency(
                        func, elapsed_time, logger, log_level, include_args, args, kwargs
                    )

            return async_gen_wrapper  # type: ignore
        elif is_sync_gen:
            # Handle sync generator functions
            @functools.wraps(func)
            def sync_gen_wrapper(*args, **kwargs) -> Generator:
                start_time = time.perf_counter()
                try:
                    for item in func(*args, **kwargs):
                        yield item
                finally:
                    elapsed_time = time.perf_counter() - start_time
                    _log_latency(
                        func, elapsed_time, logger, log_level, include_args, args, kwargs
                    )

            return sync_gen_wrapper  # type: ignore
        elif is_async:
            # Handle regular async functions
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed_time = time.perf_counter() - start_time
                    _log_latency(
                        func, elapsed_time, logger, log_level, include_args, args, kwargs
                    )
                    # Store latency as attribute on result if it's an object
                    if hasattr(result, '__dict__'):
                        setattr(result, '_latency', elapsed_time)

            return async_wrapper  # type: ignore
        else:
            # Handle regular sync functions
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed_time = time.perf_counter() - start_time
                    _log_latency(
                        func, elapsed_time, logger, log_level, include_args, args, kwargs
                    )
                    # Store latency as attribute on result if it's an object
                    if hasattr(result, '__dict__'):
                        setattr(result, '_latency', elapsed_time)

            return sync_wrapper  # type: ignore

    # Support both @measure_latency and @measure_latency() usage
    if func is not None:
        # Called without parentheses: @measure_latency
        return decorator(func)
    else:
        # Called with parentheses: @measure_latency() or @measure_latency(logger=...)
        return decorator


def _log_latency(
    func: Callable,
    elapsed_time: float,
    logger: Optional[Callable],
    log_level: str,
    include_args: bool,
    args: tuple,
    kwargs: dict,
) -> None:
    """Helper function to log latency information."""


    # Format elapsed time
    if elapsed_time < ONE_MILLISECOND:
        time_str = f"{elapsed_time * 1_000_000:.2f}μs"
    elif elapsed_time < ONE_SECOND:
        time_str = f"{elapsed_time * 1000:.2f}ms"
    else:
        time_str = f"{elapsed_time:.2f}s"

    # Build log message
    func_name = func.__name__
    message = f"Function '{func_name}' executed in {time_str}"

    if include_args:
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        if params:
            message += f" with args: {params}"

    # Log with appropriate level
    log_method = getattr(_logger, log_level.lower(), None)
    if log_method and callable(log_method):
        log_method(message)
    elif hasattr(_logger, 'info'):
        _logger.info(message)  # Fallback to info if level not found

