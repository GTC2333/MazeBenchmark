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
    for k in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','model','output_dir']:
        if os.getenv(k):
            cfg[k] = os.getenv(k)
    return cfg


def apply_env_keys(cfg: Dict):
    for k in ['OPENAI_API_KEY','ANTHROPIC_API_KEY']:
        v = cfg.get(k)
        if v:
            os.environ[k] = v
