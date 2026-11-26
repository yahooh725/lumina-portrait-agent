#!/usr/bin/env python3
"""Lumina 拾光人像画室一键启动入口。

用法示例：
  python3 app.py                       # 自动检测环境（有 GPU + LoRA 启用多模态模型，否则使用规则规划器）
  python3 app.py --mode rules          # 纯 CPU 模式（中文规则规划 + 原生六工具像素精修引擎）
  python3 app.py --mode lora --gpu 0   # GPU 模式（Qwen3-VL-2B + 内置六工具 LoRA）
  python3 app.py --mode mock           # 极速交互预览模式（不加载视觉/大模型）
  python3 app.py --preflight           # 环境自检
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.planners import resolve_default_adapter, resolve_default_base_model
from src.web.server import preflight_summary, run_server


def _detect_auto_mode() -> str:
    adapter_ok = (resolve_default_adapter() / 'adapter_model.safetensors').is_file()
    if not adapter_ok:
        return 'rules'
    try:
        import torch
        if torch.cuda.is_available():
            base = resolve_default_base_model()
            if Path(base).is_dir() or os.environ.get('HF_HUB_OFFLINE') != '1':
                return 'six1800'
    except Exception:
        pass
    return 'rules'


def main() -> None:
    parser = argparse.ArgumentParser(description='Lumina 拾光人像画室 — 中文对话人像修图工作室')
    parser.add_argument('--host', default=os.environ.get('PORTRAIT_DEMO_HOST', '127.0.0.1'), help='监听地址（默认 127.0.0.1）')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORTRAIT_DEMO_PORT', '8787')), help='监听端口（默认 8787）')
    parser.add_argument(
        '--mode',
        choices=['auto', 'lora', 'rules', 'mock'],
        default='auto',
        help='规划模式：auto（自动检测）、lora（Qwen3-VL+LoRA）、rules（规则规划+原生引擎）、mock（轻量预览）',
    )
    parser.add_argument('--gpu', default=None, help='指定 GPU 编号（如 0）')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'runs/web_studio', help='会话与上传缓存目录')
    parser.add_argument('--no-preload', action='store_true', help='延迟至首轮对话时再加载大模型')
    parser.add_argument('--preflight', action='store_true', help='仅检查运行资源，不启动服务')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    mode_map = {'lora': 'six1800', 'rules': 'rules', 'mock': 'mock'}
    planner_mode = _detect_auto_mode() if args.mode == 'auto' else mode_map[args.mode]

    if args.preflight:
        ok, checks = preflight_summary(planner_mode)
        print(json.dumps({'status': 'ready' if ok else 'failed', **checks}, ensure_ascii=False, indent=2))
        raise SystemExit(0 if ok else 1)

    run_server(
        host=args.host,
        port=args.port,
        planner_mode=planner_mode,
        data_dir=args.data_dir,
        preload=(planner_mode == 'six1800' and not args.no_preload),
    )


if __name__ == '__main__':
    main()
