"""PortraitAgent 可插拔规划器（六工具 LoRA / 规则规划 / Mock 测试）。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'agent_baseline'
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import app as legacy_app

DEFAULT_MODEL_DIR = BASE / 'models/Qwen3-VL-2B-Instruct'
DEFAULT_ADAPTER_DIR = ROOT / 'weights/six_tool_lora_v1'
LEGACY_ADAPTER_DIR = BASE / 'tool_expand_v1/six_tool_data_v1/train_prep_v1/sft_from_base_1800_v2/adapter'
DEFAULT_PROMPT_PATH = ROOT / 'configs/planner_prompt.txt'
LEGACY_PROMPT_PATH = BASE / 'tool_expand_v1/six_tool_data_v1/eval_dev280_v1/prompt_snapshot.txt'


def resolve_default_prompt() -> Path:
    return DEFAULT_PROMPT_PATH if DEFAULT_PROMPT_PATH.is_file() else LEGACY_PROMPT_PATH


def resolve_default_adapter() -> Path:
    env_adapter = os.environ.get('PORTRAIT_SIX_TOOL_ADAPTER')
    if env_adapter:
        return Path(env_adapter).expanduser().resolve()
    return (DEFAULT_ADAPTER_DIR if DEFAULT_ADAPTER_DIR.is_dir() else LEGACY_ADAPTER_DIR).resolve()


def resolve_default_base_model() -> str:
    env_model = os.environ.get('PORTRAIT_BASE_MODEL')
    if env_model:
        return env_model
    if DEFAULT_MODEL_DIR.is_dir():
        return str(DEFAULT_MODEL_DIR)
    return 'Qwen/Qwen3-VL-2B-Instruct'


def parse_json_plan(raw: str) -> tuple[dict | None, bool]:
    """解析模型输出 JSON；返回 (parsed_dict, is_strict_pure_json)。"""
    try:
        parsed = json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        return (parsed if isinstance(parsed, dict) else None), isinstance(parsed, dict)
    except (ValueError, json.JSONDecodeError):
        pass
    match = re.search(r'\{.*\}', raw, re.S)
    if match:
        try:
            parsed = json.loads(match.group(0), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            return (parsed if isinstance(parsed, dict) else None), False
        except (ValueError, json.JSONDecodeError):
            pass
    return None, False


@dataclass(frozen=True)
class PlanResult:
    raw_text: str
    strict_json: bool
    parsed_plan: dict | None
    plan_seconds: float
    model_name: str


class BasePlanner:
    name: str = 'base'

    def preload(self) -> dict:
        return {'planner': self.name, 'loaded': True}

    def is_loaded(self) -> bool:
        return True

    def metadata(self) -> dict:
        return {'planner_mode': self.name, 'model_loaded': self.is_loaded()}

    def plan(self, llm_context: dict, instruction: str) -> PlanResult:
        raise NotImplementedError


class RulePlanner(BasePlanner):
    name = 'rules'

    def plan(self, llm_context: dict, instruction: str) -> PlanResult:
        t0 = time.monotonic()
        plan = legacy_app.rule_plan(instruction, llm_context['state'], llm_context['dialogue'])
        return PlanResult(
            raw_text=json.dumps(plan, ensure_ascii=False),
            strict_json=True,
            parsed_plan=plan,
            plan_seconds=round(time.monotonic() - t0, 4),
            model_name='rules',
        )


class MockPlanner(BasePlanner):
    name = 'mock'

    def __init__(self, override_fn: Callable[[dict, str], dict] | None = None):
        self._override = override_fn

    def plan(self, llm_context: dict, instruction: str) -> PlanResult:
        t0 = time.monotonic()
        plan = (
            self._override(llm_context, instruction)
            if self._override is not None
            else legacy_app.rule_plan(instruction, llm_context['state'], llm_context['dialogue'])
        )
        return PlanResult(
            raw_text=json.dumps(plan, ensure_ascii=False),
            strict_json=True,
            parsed_plan=plan,
            plan_seconds=round(time.monotonic() - t0, 4),
            model_name='mock',
        )


class SixToolLoraPlanner(BasePlanner):
    """封装 Qwen3-VL-2B-Instruct + 六工具语言侧 LoRA 规划器。"""

    name = 'six1800'

    def __init__(
        self,
        prompt_path: Path | None = None,
        adapter_path: Path | None = None,
        base_model: str | Path | None = None,
    ):
        self.prompt_path = (prompt_path or resolve_default_prompt()).resolve()
        self.prompt_text = self.prompt_path.read_text(encoding='utf-8')
        self.adapter_path = (adapter_path.resolve() if adapter_path else resolve_default_adapter())
        self.base_model = str(base_model) if base_model is not None else resolve_default_base_model()
        self._bundle = None
        self._load_seconds: float | None = None
        self._env: dict | None = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self._bundle is not None

    def preload(self) -> dict:
        with self._lock:
            if self._bundle is None:
                import peft
                import torch
                import transformers
                from peft import PeftModel
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

                t0 = time.monotonic()
                if not torch.cuda.is_available():
                    raise SystemExit(f'未检测到可用 GPU；CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')
                device = torch.device('cuda:0')
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                local_only = Path(self.base_model).is_dir() or os.environ.get('HF_HUB_OFFLINE') == '1'
                processor = AutoProcessor.from_pretrained(self.base_model, local_files_only=local_only)
                base = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.base_model,
                    dtype=dtype,
                    device_map={'': 0},
                    local_files_only=local_only,
                    attn_implementation='sdpa',
                )
                model = PeftModel.from_pretrained(base, str(self.adapter_path), is_trainable=False)
                model.eval()
                self._load_seconds = round(time.monotonic() - t0, 3)
                self._env = {
                    'gpu': torch.cuda.get_device_name(device),
                    'dtype': str(dtype),
                    'torch': torch.__version__,
                    'transformers': transformers.__version__,
                    'peft': peft.__version__,
                }
                self._bundle = (model, processor, torch, device)
        return {
            'planner': self.name,
            'loaded': True,
            'load_seconds': self._load_seconds,
            'environment': self._env,
            'adapter': str(self.adapter_path),
            'base_model': self.base_model,
        }

    def metadata(self) -> dict:
        return {
            'planner_mode': self.name,
            'model_loaded': self.is_loaded(),
            'model_load_seconds': self._load_seconds,
            'environment': self._env,
            'adapter': str(self.adapter_path),
            'base_model': self.base_model,
        }

    def plan(self, llm_context: dict, instruction: str) -> PlanResult:
        self.preload()
        content = (
            self.prompt_text
            .replace('{state}', json.dumps(llm_context['state'], ensure_ascii=False))
            .replace('{history}', json.dumps(llm_context['dialogue'][-8:], ensure_ascii=False))
            .replace('{recent_edit}', json.dumps(llm_context.get('recent_edit'), ensure_ascii=False))
            .replace('{pending_question}', json.dumps((llm_context.get('open_signal') or {}).get('question'), ensure_ascii=False))
            .replace('{context_hint}', json.dumps(None, ensure_ascii=False))
            .replace('{instruction}', instruction)
        )
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': llm_context['original']},
                {'type': 'text', 'text': content},
            ],
        }]
        with self._lock:
            model, processor, torch, device = self._bundle
            started = time.monotonic()
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors='pt',
            ).to(device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            torch.cuda.synchronize(device)
            raw = processor.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            del output, inputs
            plan_seconds = round(time.monotonic() - started, 3)

        parsed, strict = parse_json_plan(raw)
        return PlanResult(
            raw_text=raw,
            strict_json=strict,
            parsed_plan=parsed,
            plan_seconds=plan_seconds,
            model_name='six1800',
        )
