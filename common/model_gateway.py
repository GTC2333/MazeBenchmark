import os
import base64
import requests
from typing import Optional, Dict, Any


class MockAdapter:
    def __init__(self, model: str = 'mock'):
        self._model = model

    def name(self) -> str:
        return f'mock:{self._model}'

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        import re
        m = re.search(r"(\d+)x(\d+)", prompt)
        h, w = (10, 10)
        if m:
            h, w = int(m.group(1)), int(m.group(2))
        path = []
        x, y = 0, 0
        path.append((x, y))
        while y < w - 1:
            y += 1
            path.append((x, y))
        while x < h - 1:
            x += 1
            path.append((x, y))
        return str(path)


class ChatAdapter:
    """
    Unified chat adapter supporting provider in {'openai', 'azure'} and mock.
    Implements a single generate(prompt, image_path=None) surface used by both
    text and image flows.
    """
    def __init__(
        self,
        provider: str = 'openai',
        model: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        use_sdk: Optional[bool] = None,
    ):
        self.provider = (provider or 'openai').lower()
        self.model = model
        self.api_key = api_key or (
            os.getenv('AZURE_OPENAI_API_KEY') if self.provider == 'azure' else os.getenv('OPENAI_API_KEY')
        ) or ''
        # For OpenAI, api_base defaults to public endpoint; for Azure it's the endpoint root (e.g., https://xxx.openai.azure.com)
        if self.provider == 'azure':
            self.endpoint = api_base or os.getenv('AZURE_OPENAI_ENDPOINT') or ''
            self.deployment = deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT') or model
            self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION') or '2024-08-01-preview'
        else:
            self.api_base = api_base or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
        self.use_sdk = True if use_sdk is None else bool(use_sdk)

    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def _build_messages(self, prompt: str, image_path: Optional[str]) -> Any:
        # Text-only message
        if not image_path:
            return [
                {'role': 'system', 'content': '只输出纯文本坐标路径，如 [(0,0),(0,1),...]，禁止解释。'},
                {'role': 'user', 'content': prompt},
            ]
        # Multimodal message for image + text
        content = [{'type': 'text', 'text': prompt}]
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}})
        return [
            {'role': 'system', 'content': '只输出纯文本坐标路径，如 [(0,0),(0,1),...]，禁止解释。'},
            {'role': 'user', 'content': content},
        ]

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        messages = self._build_messages(prompt, image_path)
        if self.provider == 'mock':
            return MockAdapter().generate(prompt, image_path)
        if self.provider == 'openai':
            # Try SDK first if requested
            if self.use_sdk:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=self.api_key, base_url=self.api_base)
                    comp = client.chat.completions.create(model=self.model, messages=messages, temperature=0.0)
                    return comp.choices[0].message.content
                except Exception:
                    pass
            # HTTP fallback
            url = f"{self.api_base}/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            data = {"model": self.model, "messages": messages, "temperature": 0.0}
            resp = requests.post(url, headers=headers, json=data, timeout=15)
            resp.raise_for_status()
            j = resp.json()
            return j['choices'][0]['message']['content']
        elif self.provider == 'azure':
            # Azure OpenAI uses deployment in path and api-version query; header is api-key
            if not self.endpoint:
                raise RuntimeError('AZURE endpoint not configured')
            url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
            headers = {"api-key": self.api_key, "Content-Type": "application/json"}
            data: Dict[str, Any] = {"messages": messages, "temperature": 0.0}
            # In Azure, model name is generally the deployment name
            resp = requests.post(url, headers=headers, json=data, timeout=20)
            resp.raise_for_status()
            j = resp.json()
            return j['choices'][0]['message']['content']
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


def make_adapter_from_cfg(cfg: Dict[str, Any], image: bool = False) -> ChatAdapter | MockAdapter:
    model = cfg.get('model', 'mock')
    provider = (cfg.get('PROVIDER') or '').lower()
    # Heuristics: explicit provider > model prefix > presence of azure keys
    if not provider:
        if model.startswith('azure:'):
            provider = 'azure'
            model = model.split(':', 1)[1]
        elif model.startswith('mock'):
            provider = 'mock'
        elif os.getenv('AZURE_OPENAI_API_KEY') or cfg.get('AZURE_OPENAI_API_KEY'):
            provider = 'azure'
        else:
            provider = 'openai'
    # Resolve API key for OpenAI-compatible providers
    def _resolve_openai_key() -> str:
        if cfg.get('OPENAI_API_KEY'):
            return str(cfg.get('OPENAI_API_KEY'))
        env_name = cfg.get('OPENAI_API_KEY_ENV') or os.getenv('OPENAI_API_KEY_ENV')
        if env_name:
            # Prefer environment variable value; fallback to config entry with that name
            return os.getenv(env_name) or str(cfg.get(env_name) or '')
        return os.getenv('OPENAI_API_KEY') or ''

    if provider == 'mock' or model.startswith('mock'):
        return MockAdapter(model=model)

    if provider == 'azure':
        ak = cfg.get('AZURE_OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY') or ''
        if not ak:
            return MockAdapter(model=model)
        return ChatAdapter(
            provider='azure',
            model=model,
            api_key=ak,
            api_base=cfg.get('AZURE_OPENAI_ENDPOINT') or os.getenv('AZURE_OPENAI_ENDPOINT'),
            deployment=cfg.get('AZURE_OPENAI_DEPLOYMENT') or os.getenv('AZURE_OPENAI_DEPLOYMENT') or model,
            api_version=cfg.get('AZURE_OPENAI_API_VERSION') or os.getenv('AZURE_OPENAI_API_VERSION') or '2024-08-01-preview',
            use_sdk=False,  # requests path is more reliable for Azure here
        )

    # default openai-compatible
    use_sdk = (cfg.get('USE_OPENAI_SDK') in ('1', 'true', 'True')) if cfg.get('USE_OPENAI_SDK') is not None else (os.getenv('USE_OPENAI_SDK') in ('1','true','True'))
    ok = _resolve_openai_key()
    if not ok:
        return MockAdapter(model=model)
    return ChatAdapter(
        provider='openai',
        model=model,
        api_key=ok,
        api_base=cfg.get('OPENAI_API_BASE') or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1',
        use_sdk=use_sdk if use_sdk is not None else True,
    )
