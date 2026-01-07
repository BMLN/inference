# src/inference/providers/__init__.py
try:
    import torch  # optional dependency
except ImportError as e:
    raise ImportError(
        "The 'inference.local' subpackage requires extra dependencies.\n"
        "Install them with: pip install inference[local]"
    ) from e
