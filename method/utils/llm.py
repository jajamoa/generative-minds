"""
Shared LLM utilities for context extraction (Qwen integration).

Provides:
- BaseContextLLM: abstract interface
- ContextLLMConfig: YAML-driven configuration
- QwenMaxContextLLM: requests-based client

Prompt modules are expected under either:
- method/prompt/<version>.py  (preferred)
- method/motif_library/prompt/<version>.py (fallback)

Configs are expected under either:
- method/configs/*.yaml (preferred)
- method/motif_library/configs/*.yaml (fallback)
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional until LLM is used
    yaml = None


# Load env from repo/method root
_here = Path(__file__).resolve()
_method_root = _here.parent.parent
load_dotenv(dotenv_path=_method_root / ".env")


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _resolve_config_path(cfg_name_or_path: str) -> Path:
    """Resolve a config filename or absolute path to a Path.

    Search order:
    - absolute/relative path as given if exists
    - method/configs/<name>
    - method/motif_library/configs/<name>
    """
    p = Path(cfg_name_or_path)
    if p.exists():
        return p
    candidates = [
        _method_root / "configs" / cfg_name_or_path,
        _method_root / "configs" / "context_graph" / cfg_name_or_path,
        _method_root / "motif_library" / "configs" / cfg_name_or_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Config not found: {cfg_name_or_path}")


def _load_prompt_module(version: str):
    """Load prompt module by version from preferred or fallback locations."""
    candidates = [
        _method_root / "prompt" / f"{version}.py",
        _method_root / "motif_library" / "prompt" / f"{version}.py",
    ]
    for path in candidates:
        if path.exists():
            import importlib.util as iutil

            spec = iutil.spec_from_file_location(version, path)
            module = iutil.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Prompt module not found for version '{version}'")


@dataclass
class ContextLLMConfig:
    """LLM configuration loaded from YAML for context extraction."""

    config_file: str = "qwen_context_default.yaml"

    # Populated in __post_init__
    model_name: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024
    base_url: str = ""
    timeout: int = 60
    prompt_version: str = "context_v1"
    prompt_template: str | None = None
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if yaml is None:
            raise ImportError("PyYAML is required to load LLM configs. Install pyyaml.")

        path = _resolve_config_path(self.config_file)
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_cfg = cfg.get("model", {})
        self.model_name = model_cfg.get("name", self.model_name)
        self.temperature = model_cfg.get("temperature", self.temperature)
        self.max_tokens = int(model_cfg.get("max_tokens", self.max_tokens))

        api_cfg = cfg.get("api", {})
        self.base_url = api_cfg.get("base_url", self.base_url)
        self.timeout = int(api_cfg.get("timeout", self.timeout))

        prompt_cfg = cfg.get("prompt", {})
        self.prompt_version = prompt_cfg.get("version", self.prompt_version)
        tmpl = prompt_cfg.get("template")
        if isinstance(tmpl, str) and tmpl.strip():
            self.prompt_template = tmpl
        sys_tmpl = prompt_cfg.get("system_prompt")
        if isinstance(sys_tmpl, str) and sys_tmpl.strip():
            self.system_prompt = sys_tmpl


class BaseContextLLM(ABC):
    """Abstract interface for extracting context factors and causal pairs from text."""

    @abstractmethod
    def generate_context(
        self, post_title: str, scenario: str, comment: str, stance: str
    ) -> Dict[str, Any]:
        """Return {factors: List[str|{name,evidence}], edges?: [{from,to,relation,cue?}], causal_pairs?: [{cause,effect,cue?}], _raw?: str, _prompt?: str}."""
        raise NotImplementedError


class QwenMaxContextLLM(BaseContextLLM):
    """Qwen Max client for context extraction via HTTP requests."""

    def __init__(self, config: ContextLLMConfig) -> None:
        self.cfg = config
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("QWEN_API_KEY not found in environment variables")
        self.prompt_module = None
        if not self.cfg.prompt_template:
            self.prompt_module = _load_prompt_module(self.cfg.prompt_version)

    def _create_prompt(
        self, post_title: str, scenario: str, comment: str, stance: str
    ) -> str:
        if self.cfg.prompt_template:
            prompt = self.cfg.prompt_template
            # Avoid .format to preserve JSON braces in template; use simple replacement
            replacements = {
                "{post_title}": post_title,
                "{scenario}": scenario,
                "{comment}": comment,
                "{stance}": stance,
            }
            for k, v in replacements.items():
                prompt = prompt.replace(k, v)
            return prompt
        assert (
            self.prompt_module is not None
        ), "Prompt module missing and no template provided"
        return self.prompt_module.create_prompt(post_title, scenario, comment, stance)

    def generate_context(
        self, post_title: str, scenario: str, comment: str, stance: str
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        prompt = self._create_prompt(post_title, scenario, comment, stance)
        messages = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.cfg.model_name,
            "input": {"messages": messages},
            "parameters": {
                "temperature": self.cfg.temperature,
                "result_format": "message",
                "max_tokens": self.cfg.max_tokens,
            },
        }

        try:
            resp = requests.post(
                self.cfg.base_url,
                headers=headers,
                json=payload,
                timeout=self.cfg.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            content = result["output"]["choices"][0]["message"]["content"]

            raw = content
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                parsed: Dict[str, Any] = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Qwen JSON parse failed, attempting repair: {e}")
                lines = [
                    ln for ln in content.split("\n") if not ln.strip().startswith("//")
                ]
                content2 = "\n".join(lines)
                try:
                    parsed = json.loads(content2)
                    logger.info("Qwen JSON repair successful")
                except json.JSONDecodeError as e2:
                    logger.error(f"Qwen JSON repair failed: {e2}")
                    return {
                        "factors": [],
                        "causal_pairs": [],
                        "_raw": raw,
                        "_prompt": prompt,
                    }

            factors = parsed.get("factors") or parsed.get("aspects") or []
            if not isinstance(factors, list):
                factors = []
            edges = parsed.get("edges") or []
            if not isinstance(edges, list):
                edges = []
            causal_pairs = parsed.get("causal_pairs") or parsed.get("links") or []
            if not isinstance(causal_pairs, list):
                causal_pairs = []

            return {
                "factors": factors,
                "edges": edges,
                "causal_pairs": causal_pairs,
                "_raw": raw,
                "_prompt": prompt,
            }
        except Exception as e:
            logger.error(f"Qwen API call failed: {e}")
            raise


def build_llm_from_cfg(cfg_name_or_path: str) -> QwenMaxContextLLM:
    """Helper to instantiate Qwen LLM from a config filename or path."""
    cfg = ContextLLMConfig(config_file=cfg_name_or_path)
    return QwenMaxContextLLM(cfg)
