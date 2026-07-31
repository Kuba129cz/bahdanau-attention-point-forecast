# meteo/decorators.py
from typing import Callable, Any, Set, Union
import functools
import requests

def handle_api_errors(label: str):
    """
    Remove try-except blocks from site's requests
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except requests.RequestException as e:
                status = e.response.status_code if e.response else "No Response"
                raise RuntimeError(f"{label} API failed (Status: {status}): {e}")
            except ValueError as e:
                raise RuntimeError(f"Invalid JSON received from {label} API: {e}")
            
        return wrapper
    return decorator

def validate_output(schema: Set[str], expected_count: Union[int, Callable[..., int], None] = None, label: str = ""):
    """
    Guarantees that the function returns complete data with valid schema and row counts.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            data = func(*args, **kwargs)
            if not data:
                raise ValueError(f"{label} data is empty.")
            
            missing = schema - data.keys()
            if missing:
                raise KeyError(f"{label} missing columns: {missing}")
            
            if callable(expected_count):
                target_count = expected_count(*args, **kwargs)
            else:
                target_count = expected_count
            
            sample_key = next(iter(schema))
            actual_count = len(data[sample_key])
            
            if target_count is not None and actual_count != target_count:
                raise ValueError(f"{label} validation failed: Expected {target_count} records, got {actual_count}.")
            
            for key in schema:
                if len(data[key]) != actual_count:
                    raise ValueError(f"{label} column '{key}' length mismatch.")
                if any(v is None for v in data[key]):
                    raise ValueError(f"{label} '{key}' contains NULL.")
            
            return data
        return wrapper
    return decorator