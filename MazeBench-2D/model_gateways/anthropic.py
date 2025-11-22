import os
import requests
from .base import ModelAdapter

class AnthropicAdapter(ModelAdapter):
    def __init__(self, model: str = 'claude-3-5-sonnet-20241022', api_base: str = 'https://api.anthropic.com/v1'):
        self._model = model
        self.api_base = api_base
        self.api_key = os.getenv('ANTHROPIC_API_KEY', '')

    def name(self) -> str:
        return f'anthropic:{self._model}'

    def generate(self, prompt: str) -> str:
        url = f"{self.api_base}/messages"
        headers = {
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        data = {
            'model': self._model,
            'max_tokens': 512,
            'system': '输出纯文本路径，如 [(0,0),(0,1),...]，禁止解释。',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.0
        }
        resp = requests.post(url, headers=headers, json=data, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        # Claude returns content list
        content = j['content'][0]['text'] if j.get('content') else ''
        return content
