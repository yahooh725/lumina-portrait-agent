#!/usr/bin/env python3
"""One-epoch, six-tool plan-only Qwen3-VL LoRA from base on 1968 targeted samples (v2)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / 'agent_baseline/models/Qwen3-VL-2B-Instruct'
DATA = HERE / 'train_plan_only_1968.jsonl'
FREEZE = HERE / 'train_manifest_v2.json'
PROMPT = HERE / 'prompt_snapshot.txt'
RUN = HERE / 'sft_from_base_1968_v1'
OLD_RUN_MANIFEST = ROOT / 'agent_baseline/tool_expand_v1/six_tool_data_v1/train_prep_v1/sft_from_base_1800_v2/run_manifest.json'
DEV = ROOT / 'agent_baseline/tool_expand_v1/six_tool_data_v1/eval_dev280_v1/dev280_diagnostic_frozen.jsonl'

# Identical hyperparameters to v1 (sft_from_base_1800_v2)
SEED = 20261122
MAX_LENGTH = 3072
GRAD_ACCUM = 4
LR = 2e-4


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true', help='Verify frozen v2 data without loading GPU model')
    args = parser.parse_args()

    freeze = json.loads(FREEZE.read_text(encoding='utf-8'))
    if (
        freeze['status'] != 'qualified_for_six_tool_plan_only_sft_v2'
        or sha(DATA) != freeze['frozen_train_sha256']
        or sha(PROMPT) != freeze['prompt_sha256']
        or sha(DEV) != freeze['dev_diagnostic_sha256']
        or freeze['max_length'] != MAX_LENGTH
    ):
        raise SystemExit('Frozen v2 training data, prompt, or dev280 changed')

    cases = [json.loads(line) for line in DATA.open(encoding='utf-8') if line.strip()]
    if len(cases) != 1968 or len({r['sample_id'] for r in cases}) != 1968 or len({r['image_id'] for r in cases}) != 199:
        raise SystemExit('Train v2 counts mismatch')
    for row in cases:
        if row['project_split'] != 'train' or row['eligible_for_training'] is not True:
            raise SystemExit(f"Unqualified row: {row['sample_id']}")

    old_run = json.loads(OLD_RUN_MANIFEST.read_text(encoding='utf-8'))
    if (
        sha(MODEL / 'config.json') != old_run['model_config_sha256']
        or sha(MODEL / 'model.safetensors') != old_run['model_weights_sha256']
    ):
        raise SystemExit('Base model differs from v1 training')
    if RUN.exists():
        raise SystemExit(f'Refusing to overwrite existing run: {RUN}')

    if args.preflight:
        print(json.dumps({
            'status': 'ready',
            'samples': len(cases),
            'optimizer_steps': len(cases) // GRAD_ACCUM,
            'images': len({r['image_id'] for r in cases}),
            'data_sha256': sha(DATA),
            'max_length': MAX_LENGTH,
            'formal_training_started': False,
        }, ensure_ascii=False))
        return

    import peft
    from peft import LoraConfig, get_peft_model
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('Expected one visible CUDA GPU')
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.Random(SEED).shuffle(cases)

    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL), dtype=dtype, device_map={'': 0}, local_files_only=True, attn_implementation='sdpa'
    )
    base.requires_grad_(False)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias='none',
        target_modules=r'model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)',
    )
    model = get_peft_model(base, lora)
    model.train()

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('language_model' not in name or 'visual' in name for name, _ in trainable):
        raise SystemExit('Unexpected trainable LoRA module')
    if any(p.requires_grad for p in model.base_model.model.model.visual.parameters()):
        raise SystemExit('Visual encoder is trainable')

    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=LR)
    RUN.mkdir()
    started = time.monotonic()

    run_manifest = {
        'purpose': 'one-epoch six-tool plan-only LoRA SFT from base on 1968 samples (1800 original + 168 targeted contrast/recovery)',
        'starting_checkpoint': 'base_model_fresh_lora',
        'seed': SEED,
        'data_sha256': sha(DATA),
        'freeze_sha256': sha(FREEZE),
        'dev_diagnostic_sha256': sha(DEV),
        'script_sha256': sha(Path(__file__)),
        'prompt_sha256': sha(PROMPT),
        'model_config_sha256': sha(MODEL / 'config.json'),
        'model_weights_sha256': sha(MODEL / 'model.safetensors'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_name': gpu_name,
        'dtype': str(dtype),
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
        'epochs': 1,
        'examples': len(cases),
        'gradient_accumulation': GRAD_ACCUM,
        'max_length': MAX_LENGTH,
        'learning_rate': LR,
        'lora': {'r': 4, 'alpha': 8, 'dropout': 0.0, 'target_modules': lora.target_modules},
        'trainable_parameter_count': sum(p.numel() for _, p in trainable),
        'visual_parameters_trainable': 0,
        'test_used': False,
    }
    (RUN / 'run_manifest.json').write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    torch.cuda.reset_peak_memory_stats(0)
    examples = 0
    supervised_tokens = 0
    optimizer_steps = 0
    losses = []

    with (RUN / 'progress.jsonl').open('w', encoding='utf-8', buffering=1) as progress:
        for start in range(0, len(cases), GRAD_ACCUM):
            group = cases[start:start + GRAD_ACCUM]
            optimizer.zero_grad(set_to_none=True)
            group_losses = []
            for row in group:
                prefix = processor.apply_chat_template(
                    row['messages'], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt'
                )
                messages = row['messages'] + [{'role': 'assistant', 'content': row['completion']}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt'
                )
                pids, fids = prefix['input_ids'][0], inputs['input_ids'][0]
                prompt_len = len(pids)
                if prompt_len >= len(fids) or len(fids) > MAX_LENGTH or not (pids == fids[:prompt_len]).all().item():
                    raise SystemExit(f"Token boundary/length changed: {row['sample_id']}")
                labels = inputs['input_ids'].clone()
                labels[:, :prompt_len] = -100
                inputs.pop('token_type_ids', None)
                inputs = inputs.to(device)
                inputs['labels'] = labels.to(device)
                output = model(**inputs)
                if output.loss is None or not torch.isfinite(output.loss).item():
                    raise SystemExit(f"Non-finite loss: {row['sample_id']}")
                val = output.loss.item()
                (output.loss / len(group)).backward()
                group_losses.append(val)
                losses.append(val)
                examples += 1
                supervised_tokens += len(fids) - prompt_len
                del output, inputs, prefix
            optimizer.step()
            optimizer_steps += 1
            item = {
                'optimizer_step': optimizer_steps,
                'examples_seen': examples,
                'group_loss_mean': sum(group_losses) / len(group_losses),
                'elapsed_seconds': round(time.monotonic() - started, 2),
            }
            progress.write(json.dumps(item, ensure_ascii=False) + '\n')
            if optimizer_steps % 10 == 0 or examples == len(cases):
                print(json.dumps(item, ensure_ascii=False), flush=True)

    adapter = RUN / 'adapter'
    model.save_pretrained(str(adapter), safe_serialization=True)
    torch.cuda.synchronize(device)

    summary = {
        'status': 'complete',
        'examples_seen': examples,
        'optimizer_steps': optimizer_steps,
        'supervised_tokens': supervised_tokens,
        'first_10_loss_mean': sum(losses[:10]) / min(10, len(losses)),
        'last_10_loss_mean': sum(losses[-10:]) / min(10, len(losses)),
        'mean_loss': sum(losses) / len(losses),
        'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'adapter_config_sha256': sha(adapter / 'adapter_config.json'),
        'adapter_weights_sha256': sha(adapter / 'adapter_model.safetensors'),
        'evaluation_performed': False,
    }
    (RUN / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
