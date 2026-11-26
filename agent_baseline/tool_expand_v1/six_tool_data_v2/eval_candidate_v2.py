#!/usr/bin/env python3
"""Evaluate six-tool v2 adapter (1968 samples) on frozen dev280 + 12-case clarify-recovery regression.

Enforces the pre-registered selection gate from second_round_decision.md:
  1. dev280 strict plan_ok > 254 (v1 baseline = 254/280)
  2. dev280 state_equiv_ok > 256 (v1 baseline = 256/280)
  3. old200 >= 178 AND new80 >= 76
  4. strict_json == 280 AND backend_accepted == 280
Also reports paired diff vs v1 (sft_from_base_1800_v2) and 12-case clarify-recovery regression.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / 'agent_baseline/models/Qwen3-VL-2B-Instruct'
V2_RUN = HERE / 'sft_from_base_1968_v1'
V2_ADAPTER = V2_RUN / 'adapter'
V1_RAW = ROOT / 'agent_baseline/tool_expand_v1/six_tool_data_v1/eval_dev280_v1/results/six1800/raw_results.jsonl'
DEV = ROOT / 'agent_baseline/tool_expand_v1/six_tool_data_v1/eval_dev280_v1/dev280_diagnostic_frozen.jsonl'
REGRESSION = HERE / 'regression_multiturn_12.jsonl'
PROMPT = HERE / 'prompt_snapshot.txt'
OUT = HERE / 'eval_v2_results'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'agent_baseline/tool_expand_v1/six_tool_data_v1/eval_dev280_v1'))
import compare as dev_compare
from src.tools.protocol import validate_plan_structure


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state_after_update(state: dict, plan: dict) -> dict:
    nxt = copy.deepcopy(state)
    for item in plan['updates']:
        tool = item['tool']
        nxt[tool] = {k: v for k, v in item.items() if k != 'tool'}
    return nxt


def state_equiv_ok(actual: dict | None, expected: dict, case: dict, strict_ok: bool) -> bool:
    if strict_ok:
        return True
    if not isinstance(actual, dict) or actual.get('action') != 'update' or expected.get('action') != 'update':
        return False
    try:
        validate_plan_structure(actual)
    except ValueError:
        return False
    return state_after_update(case['current_state'], actual) == state_after_update(case['current_state'], expected)


def eval_split(cases: list[dict], model, processor, prompt: str, torch, device, out_path: Path) -> tuple[list[dict], dict]:
    rows = []
    counts = Counter()
    with out_path.open('w', encoding='utf-8', buffering=1) as stream:
        for i, case in enumerate(cases, 1):
            content = (
                prompt
                .replace('{state}', json.dumps(case['current_state'], ensure_ascii=False))
                .replace('{history}', json.dumps(case['history'], ensure_ascii=False))
                .replace('{instruction}', case['instruction'])
            )
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': str(ROOT / case['image_ref'])},
                {'type': 'text', 'text': content},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt'
            ).to(device)
            t0 = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            torch.cuda.synchronize(device)
            raw = processor.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            plan, pure = dev_compare.parse(raw)
            err = None
            if plan is None:
                err = 'No parseable JSON plan'
            else:
                try:
                    validate_plan_structure(plan)
                except ValueError as exc:
                    err = str(exc)
            ok, reason = dev_compare.judge(plan, case['expected_plan'], case)
            equiv = state_equiv_ok(plan, case['expected_plan'], case, ok)
            group = 'new80' if case['sample_id'].startswith('six_v1_dev_') else 'old200'
            row = {
                'sample_id': case['sample_id'],
                'image_id': case['image_id'],
                'group': group,
                'instruction': case['instruction'],
                'expected_plan': case['expected_plan'],
                'raw_text': raw,
                'raw_pure_json': pure,
                'parsed_plan': plan,
                'raw_semantic_plan_ok': ok,
                'state_equiv_ok': equiv,
                'score_reason': reason,
                'backend_accepted': err is None,
                'backend_error': err,
                'elapsed_seconds': round(time.monotonic() - t0, 3),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows.append(row)
            counts['total'] += 1
            counts['strict_json'] += int(pure)
            counts['plan_ok'] += int(ok)
            counts['state_equiv_ok'] += int(equiv)
            counts['backend_accepted'] += int(err is None)
            counts[f'{group}_total'] += 1
            counts[f'{group}_plan_ok'] += int(ok)
            counts[f"action_{case['expected_plan']['action']}_total"] += 1
            counts[f"action_{case['expected_plan']['action']}_ok"] += int(ok)
            if i % 20 == 0 or i == len(cases):
                print(json.dumps({'split': out_path.stem, 'done': i, 'total': len(cases), 'plan_ok': counts['plan_ok']}, ensure_ascii=False), flush=True)
            del output, inputs
    return rows, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()

    dev_cases = [json.loads(line) for line in DEV.read_text(encoding='utf-8').splitlines() if line.strip()]
    reg_cases = [json.loads(line) for line in REGRESSION.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(dev_cases) == 280 and len(reg_cases) == 12
    for c in reg_cases:
        validate_plan_structure(c['expected_plan'])
        ok, reason = dev_compare.judge(c['expected_plan'], c['expected_plan'], c)
        if not ok:
            raise SystemExit(f"Regression scorer rejects label: {c['sample_id']} {reason}")

    if OUT.exists():
        raise SystemExit(f'Refusing to overwrite existing eval dir: {OUT}')
    if args.preflight:
        print(json.dumps({
            'status': 'ready',
            'dev_cases': len(dev_cases),
            'regression_cases': len(reg_cases),
            'v2_adapter_exists': (V2_ADAPTER / 'adapter_model.safetensors').is_file(),
        }, ensure_ascii=False))
        return

    if not (V2_ADAPTER / 'adapter_model.safetensors').is_file():
        raise SystemExit(f'Missing trained v2 adapter: {V2_ADAPTER}')

    import peft
    from peft import PeftModel
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('Expected one visible CUDA GPU')
    device = torch.device('cuda:0')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL), dtype=dtype, device_map={'': 0}, local_files_only=True, attn_implementation='sdpa'
    )
    model = PeftModel.from_pretrained(base, str(V2_ADAPTER), is_trainable=False)
    model.eval()

    OUT.mkdir(parents=True)
    prompt = PROMPT.read_text(encoding='utf-8')
    dev_rows, dev_counts = eval_split(dev_cases, model, processor, prompt, torch, device, OUT / 'dev280_raw.jsonl')
    reg_rows, reg_counts = eval_split(reg_cases, model, processor, prompt, torch, device, OUT / 'regression12_raw.jsonl')

    # Paired comparison against v1 (six1800)
    v1_rows = [json.loads(line) for line in V1_RAW.read_text(encoding='utf-8').splitlines() if line.strip()]
    v1_by_id = {r['sample_id']: r for r in v1_rows}
    improved = [r['sample_id'] for r in dev_rows if r['raw_semantic_plan_ok'] and not v1_by_id[r['sample_id']]['raw_semantic_plan_ok']]
    regressed = [r['sample_id'] for r in dev_rows if not r['raw_semantic_plan_ok'] and v1_by_id[r['sample_id']]['raw_semantic_plan_ok']]

    gate_checks = {
        'strict_gt_254': dev_counts['plan_ok'] > 254,
        'state_equiv_gt_256': dev_counts['state_equiv_ok'] > 256,
        'old200_ge_178': dev_counts['old200_plan_ok'] >= 178,
        'new80_ge_76': dev_counts['new80_plan_ok'] >= 76,
        'strict_json_280': dev_counts['strict_json'] == 280,
        'backend_accepted_280': dev_counts['backend_accepted'] == 280,
    }
    gate_passed = all(gate_checks.values())

    report = {
        'evaluated_utc': datetime.now(timezone.utc).isoformat(),
        'adapter_weights_sha256': sha(V2_ADAPTER / 'adapter_model.safetensors'),
        'dev280_counts': dev_counts,
        'regression12_counts': reg_counts,
        'regression12_failures': [
            {'sample_id': r['sample_id'], 'instruction': r['instruction'], 'parsed_plan': r['parsed_plan'], 'expected': r['expected_plan'], 'reason': r['score_reason']}
            for r in reg_rows if not r['raw_semantic_plan_ok']
        ],
        'vs_six1800_improved': improved,
        'vs_six1800_regressed': regressed,
        'gate_checks': gate_checks,
        'gate_passed': gate_passed,
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
    }
    (OUT / 'gate_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'dev280_plan_ok': f"{dev_counts['plan_ok']}/280",
        'dev280_state_equiv': f"{dev_counts['state_equiv_ok']}/280",
        'regression12_plan_ok': f"{reg_counts['plan_ok']}/12",
        'improved_vs_v1': len(improved),
        'regressed_vs_v1': len(regressed),
        'gate_passed': gate_passed,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
