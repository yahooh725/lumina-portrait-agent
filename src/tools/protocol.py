"""Three-channel ToolOutcome contract and executable-structure-only validator.

Inspired by Pi Durable's channel separation:
- `output`: concise text returned to the LLM context
- `structured_output`: validated machine-readable state/payload for backend logic
- `details`: rich audit/UI telemetry (applied tools, skipped target checks, latency, argv)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = json.loads((ROOT / 'configs/tool_registry.json').read_text(encoding='utf-8'))

INITIAL_STATE: dict[str, dict] = _REGISTRY['initial_state']
TOOL_FLAGS: dict[str, str] = {item['name']: item['cli_flag'] for item in _REGISTRY['tools']}
TOOL_PARAMS: dict[str, str] = {item['name']: item['parameter'] for item in _REGISTRY['tools']}
TOOL_META: list[dict] = [
    {
        'id': item['name'],
        'label': item['label'],
        'param': item['parameter'],
        'desc': item.get('note', ''),
    }
    for item in _REGISTRY['tools']
]


@dataclass(frozen=True)
class ToolOutcome:
    output: str
    structured_output: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    is_error: bool = False
    terminate: bool = True


def validate_plan_structure(plan: object) -> dict:
    """Validate executable JSON plan structure & six-tool whitelist ONLY.

    Per project contract (AGENTS.md), runtime validation NEVER uses Chinese keyword rules
    to override or mask model semantic errors.
    """
    if not isinstance(plan, dict) or set(plan) != {'action', 'updates', 'question'}:
        raise ValueError('plan must contain only action, updates, question')
    action, updates, question = plan['action'], plan['updates'], plan['question']
    if action not in ('update', 'undo', 'reset', 'clarify') or not isinstance(updates, list):
        raise ValueError('invalid action or updates')
    if action == 'clarify':
        if updates or not isinstance(question, str) or not question.strip():
            raise ValueError('clarify needs a non-empty question and no updates')
        return plan
    if question is not None or (action == 'update' and not updates) or (action != 'update' and updates):
        raise ValueError('invalid action/updates/question combination')
    if len(updates) > len(TOOL_FLAGS):
        raise ValueError('too many updates')
    seen: set[str] = set()
    for item in updates:
        if not isinstance(item, dict):
            raise ValueError('update must be an object')
        tool = item.get('tool')
        if tool not in TOOL_FLAGS or tool in seen:
            raise ValueError(f'unknown or duplicate tool: {tool}')
        seen.add(tool)
        param = TOOL_PARAMS[tool]
        if set(item) != {'tool', 'enabled', param} or type(item['enabled']) is not bool:
            raise ValueError(f'invalid fields for {tool}')
        val = item[param]
        if type(val) not in (int, float) or not 0 <= val <= 1:
            raise ValueError(f'invalid {param} for {tool}')
        if not item['enabled'] and tool != 'remove_acne' and val != 0:
            raise ValueError(f'disabled strength must be zero for {tool}')
    return plan


def merge_state(current_state: dict, history_stack: list[dict], plan: dict) -> dict:
    """Compute next six-tool parameter state from validated plan without mutating inputs."""
    action = plan['action']
    if action == 'undo':
        if not history_stack:
            raise ValueError('nothing to undo')
        return copy.deepcopy(history_stack[-1])
    if action == 'reset':
        return copy.deepcopy(INITIAL_STATE)
    if action == 'update':
        nxt = copy.deepcopy(current_state)
        for upd in plan['updates']:
            tool = upd['tool']
            nxt[tool] = {k: v for k, v in upd.items() if k != 'tool'}
        return nxt
    raise ValueError(f'cannot merge state for action: {action}')
