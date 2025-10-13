# src/inference/providers/__init__.py
try:
    import torch  # optional dependency
except ImportError as e:
    raise ImportError(
        "The 'inference.providers' subpackage requires extra dependencies.\n"
        "Install them with: pip install inference[providers]"
    ) from e
