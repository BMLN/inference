from .inference_provider import InferenceProvider
from .deepinfra import DeepInfraClient

# Die folgenden Provider nur laden, wenn Torch vorhanden ist (lokale Modelle)
try:
    import torch  # optional
    from .hf import InferenceClient as HFClient
    from .ollama import OllamaClient
except Exception:
    HFClient = None
    OllamaClient = None

__all__ = [
    "InferenceProvider",
    "DeepInfraClient",
    "HFClient",
    "OllamaClient",
]
