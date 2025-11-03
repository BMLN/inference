from .inference_provider import InferenceProvider
import os
import requests
from typing import Optional, Dict, Any, List


class DeepInfraClient(InferenceProvider):
    """
    OpenAI-kompatibles Chat-API von DeepInfra:
    POST https://api.deepinfra.com/v1/openai/chat/completions
    """

    def __init__(self, model: str, api_key: str = None, base_url: str = "https://api.deepinfra.com/v1/openai/chat/completions", **default_params):
        if not (api_key or os.getenv("DEEPINFRA_API_TOKEN")):
            raise ValueError("DeepInfra API key fehlt (setze DEEPINFRA_API_TOKEN).")
        
        self.model = model
        self.api_key = api_key or os.getenv("DEEPINFRA_API_TOKEN")
        self.url = base_url
        # Speichere alle zusätzlichen Parameter (z.B. temperature, max_tokens)
        self.default_params = default_params

        super().__init__()


    def _build_messages(self, prompt:str, system_prompt:str = None, messages=None) -> List[Dict[str, str]]:
        """
        Falls der Aufrufer schon messages übergibt, verwenden wir diese.
        Sonst bauen wir aus (optionalem) system_prompt + user prompt.
        """

        messages = messages or []

        if system_prompt:
            messages.insert(0,{"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages



    # entspricht deinem Muster: return requests.Response
    def call(self, prompt:str, *args, **kwargs):
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

        resp = requests.post(
            self.url, 
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }, 
            json=payload, 
            timeout=60
        )
        return resp



    def process_generation(self, generation_result):
        if generation_result.status_code != 200:
            # Debug-Hilfe, wie bei dir in ollama.py
            try:
                print(generation_result.json())
            except Exception:
                print(generation_result.text)
            raise Exception("DeepInfra request failed")

        # OpenAI-kompatible Struktur:
        # { choices: [ { message: { content: "..." } } ] }
        data = generation_result.json()
        data = data.get("choices")

        if not data:
            raise Exception("DeepInfra generation failed")
        

        return data[0].get("message").get("content")
