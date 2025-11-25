from pathlib import Path
from typing import Dict
import os
import yaml


def load_config() -> Dict:
    base = Path('config/config.yaml')
    local = Path('config/local.yaml')
    cfg: Dict = {}
    if base.exists():
        cfg.update(yaml.safe_load(base.read_text(encoding='utf-8')) or {})
    if local.exists():
        loc = yaml.safe_load(local.read_text(encoding='utf-8')) or {}
        cfg.update(loc)
    # Pull overrides from environment
    env_keys = [
        'OPENAI_API_KEY','ANTHROPIC_API_KEY','model','output_dir','OPENAI_API_BASE','OPENAI_API_KEY_ENV','USE_OPENAI_SDK',
        # Azure OpenAI
        'AZURE_OPENAI_API_KEY','AZURE_OPENAI_ENDPOINT','AZURE_OPENAI_DEPLOYMENT','AZURE_OPENAI_API_VERSION',
        # Provider selector
        'PROVIDER',
    ]
    for k in env_keys:
        if os.getenv(k) is not None:
            cfg[k] = os.getenv(k)
    return cfg


def apply_env_keys(cfg: Dict):
    keys = [
        'OPENAI_API_KEY','ANTHROPIC_API_KEY','OPENAI_API_BASE','OPENAI_API_KEY_ENV','USE_OPENAI_SDK',
        'AZURE_OPENAI_API_KEY','AZURE_OPENAI_ENDPOINT','AZURE_OPENAI_DEPLOYMENT','AZURE_OPENAI_API_VERSION',
        'PROVIDER',
    ]
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            os.environ[k] = str(v)
