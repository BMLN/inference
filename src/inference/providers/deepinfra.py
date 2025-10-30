from .inference_provider import InferenceProvider
import os
import requests
from typing import Optional, Dict, Any, List


class DeepInfraClient(InferenceProvider):
    """
    OpenAI-kompatibles Chat-API von DeepInfra:
    POST https://api.deepinfra.com/v1/openai/chat/completions
    """

    def __init__(
            self,
            model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
            api_key: Optional[str] = None,
            base_url: str = "https://api.deepinfra.com/v1/openai/chat/completions",
            **default_params,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPINFRA_API_TOKEN")
        if not self.api_key:
            raise ValueError("DeepInfra API key fehlt (setze DEEPINFRA_API_TOKEN).")

        self.url = base_url
        # Speichere alle zusätzlichen Parameter (z.B. temperature, max_tokens)
        self.default_params = default_params

        super().__init__()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(self, prompt: str, kwargs: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Falls der Aufrufer schon messages übergibt, verwenden wir diese.
        Sonst bauen wir aus (optionalem) system_prompt + user prompt.
        """
        if "messages" in kwargs and kwargs["messages"]:
            return kwargs["messages"]

        messages: List[Dict[str, str]] = []
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # entspricht deinem Muster: return requests.Response
    def call(self, prompt: str, *args, **kwargs):
        # ⚠️ wichtig: ctor-defaults + aufruf-kwargs zusammenführen
        params = {**self.default_params, **kwargs}

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, params),
        }

        # optionale OpenAI/DeepInfra-Parameter durchreichen
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if key in params and params[key] is not None:
                payload[key] = params[key]

        resp = requests.post(self.url, headers=self._headers(), json=payload, timeout=60)
        return resp

    def process_generation(self, generation_result):
        if generation_result.status_code != 200:
            # Debug-Hilfe, wie bei dir in ollama.py
            try:
                print(generation_result.json())
            except Exception:
                print(generation_result.text)
            raise Exception("DeepInfra request failed")

        data = generation_result.json()
        # OpenAI-kompatible Struktur:
        # { choices: [ { message: { content: "..." } } ] }
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content") or ""
