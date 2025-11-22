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

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        messages = [
            {'role': 'system', 'content': '只输出纯文本坐标路径，如 [(0,0),(0,1),...]，禁止解释。'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': prompt}
            ]}
        ]
        if image_path and os.path.exists(image_path):
            import base64
            with open(image_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            messages[1]['content'].append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}})
        data = {'model': self._model, 'messages': messages, 'temperature': 0.0}
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        return j['choices'][0]['message']['content']
