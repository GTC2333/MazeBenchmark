import os
import requests
from .base import ModelAdapter

class OpenAIAdapter(ModelAdapter):
    def __init__(self, model: str = 'gpt-4o-mini', api_base: str = 'https://api.openai.com/v1', api_key: str | None = None, api_key_env: str | None = None, use_sdk: bool | None = None):
        self._model = model
        self.api_base = os.getenv('OPENAI_API_BASE', api_base)
        # Resolve API key: explicit > env name > default OPENAI_API_KEY
        if api_key is not None:
            self.api_key = api_key
        elif api_key_env:
            self.api_key = os.getenv(api_key_env, '')
        else:
            self.api_key = os.getenv('OPENAI_API_KEY', '')
        # Prefer OpenAI SDK when available/desired; fallback to requests
        self.use_sdk = True if use_sdk is None else bool(use_sdk)

    def name(self) -> str:
        return f'openai:{self._model}'

    def generate(self, prompt: str) -> str:
        messages = [
            {'role': 'system', 'content': '输出纯文本路径，如 [(0,0),(0,1),...]，禁止解释。'},
            {'role': 'user', 'content': prompt}
        ]
        if self.use_sdk:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                completion = client.chat.completions.create(model=self._model, messages=messages, temperature=0.0)
                return completion.choices[0].message.content
            except Exception:
                # Fallback to HTTP if SDK not available or errors
                pass
        url = f"{self.api_base}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {'model': self._model, 'messages': messages, 'temperature': 0.0}
        resp = requests.post(url, headers=headers, json=data, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        return j['choices'][0]['message']['content']
