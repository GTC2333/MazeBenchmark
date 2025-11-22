import os
import requests
from .base import ModelAdapter

class OpenAIAdapter(ModelAdapter):
    def __init__(self, model: str = 'gpt-4o-mini', api_base: str = 'https://api.openai.com/v1'):
        self._model = model
        self.api_base = api_base
        self.api_key = os.getenv('OPENAI_API_KEY', '')

    def name(self) -> str:
        return f'openai:{self._model}'

    def generate(self, prompt: str) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        # Tool-less simple text response to avoid tool errors
        data = {
            'model': self._model,
            'messages': [
                {'role': 'system', 'content': '输出纯文本路径，如 [(0,0),(0,1),...]，禁止解释。'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.0
        }
        resp = requests.post(url, headers=headers, json=data, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        return j['choices'][0]['message']['content']
