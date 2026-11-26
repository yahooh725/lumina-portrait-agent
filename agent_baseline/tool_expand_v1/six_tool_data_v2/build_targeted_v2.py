#!/usr/bin/env python3
"""Build targeted v2 training set (1800 original + 168 targeted = 1968 rows) with strict anti-leakage checks.

Design:
- Keeps the original 1800 frozen training rows completely untouched.
- Adds 168 targeted contrast/recovery rows bound ONLY to the existing 199 train images.
- Verifies zero overlap with dev280 images/subject_groups and validates every completion structure.
- Also generates a 12-case diagnostic regression set for multi-turn clarify-recovery & oral decrease.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
V1_TRAIN_DIR = ROOT / 'agent_baseline/tool_expand_v1/six_tool_data_v1/train_prep_v1'
V1_TRAIN_DATA = V1_TRAIN_DIR / 'train_plan_only_1800.jsonl'
V1_PROMPT = V1_TRAIN_DIR / 'prompt_snapshot.txt'
DEV280 = ROOT / 'agent_baseline/tool_expand_v1/six_tool_data_v1/eval_dev280_v1/dev280_diagnostic_frozen.jsonl'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.tools.protocol import INITIAL_STATE, TOOL_PARAMS, validate_plan_structure

OUT_TARGETED = HERE / 'targeted_add_168.jsonl'
OUT_MERGED = HERE / 'train_plan_only_1968.jsonl'
OUT_MANIFEST = HERE / 'train_manifest_v2.json'
OUT_REVIEW_MD = HERE / 'targeted_samples_review.md'
OUT_PROMPT = HERE / 'prompt_snapshot.txt'
OUT_REGRESSION = HERE / 'regression_multiturn_12.jsonl'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_state(**overrides) -> dict:
    st = copy.deepcopy(INITIAL_STATE)
    for tool, val in overrides.items():
        if isinstance(val, tuple):
            enabled, num = val
            st[tool] = {'enabled': enabled, TOOL_PARAMS[tool]: num}
        else:
            st[tool] = {'enabled': True, TOOL_PARAMS[tool]: val}
    return st


def upd(tool: str, enabled: bool, val: float) -> dict:
    return {'action': 'update', 'updates': [{'tool': tool, 'enabled': enabled, TOOL_PARAMS[tool]: val}], 'question': None}


def clr(q: str) -> dict:
    return {'action': 'clarify', 'updates': [], 'question': q}


def build_targeted_specs() -> list[dict]:
    """Return exactly 168 targeted contrast specs across 4 gap buckets."""
    specs: list[dict] = []

    def add(bucket: str, state: dict, history: list[dict], instruction: str, plan: dict) -> None:
        validate_plan_structure(plan)
        specs.append({
            'bucket': bucket,
            'state': state,
            'history': history,
            'instruction': instruction,
            'plan': plan,
        })

    # ------------------------------------------------------------------
    # Bucket A (56 rows): Clarification recovery & interruption
    # Fixes the 0/1800 non-zero clarify->update blind spot!
    # ------------------------------------------------------------------
    # A1 (24 rows): After 1 or 2 clarify turns, user specifies target -> update
    a1_cases = [
        (make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '眼睛小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '调整大眼，眼睛小一点', upd('enlarge_eyes', True, 0.4)),
        (make_state(slim_face=0.5, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '大眼轻一点', upd('enlarge_eyes', True, 0.4)),
        (make_state(slim_face=0.5, enlarge_eyes=0.6, blur_bg=0.4),
         [{'role': 'user', 'content': '再小点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'},
          {'role': 'user', 'content': '大眼'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '我说的是大眼，减一点', upd('enlarge_eyes', True, 0.5)),
        (make_state(slim_face=0.6, smooth_skin=0.5),
         [{'role': 'user', 'content': '再轻一点'}, {'role': 'assistant', 'content': '你想减轻瘦脸还是磨皮？'}],
         '磨皮轻一点', upd('smooth_skin', True, 0.4)),
        (make_state(slim_face=0.6, smooth_skin=0.5),
         [{'role': 'user', 'content': '再轻一点'}, {'role': 'assistant', 'content': '你想减轻瘦脸还是磨皮？'}],
         '瘦脸淡一点就行', upd('slim_face', True, 0.5)),
        (make_state(remove_acne=0.5, smooth_skin=0.5),
         [{'role': 'user', 'content': '再轻一点'}, {'role': 'assistant', 'content': '你想调整祛痘还是磨皮？'}],
         '我说的是磨皮，再降一点', upd('smooth_skin', True, 0.4)),
        (make_state(remove_acne=0.5, smooth_skin=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整祛痘还是磨皮？'}],
         '祛痘敏感度减一点', upd('remove_acne', True, 0.4)),
        (make_state(blur_bg=0.6, whiten_teeth=0.5),
         [{'role': 'user', 'content': '再淡一些'}, {'role': 'assistant', 'content': '你想减轻背景虚化还是牙齿美白？'}],
         '背景虚化淡一点', upd('blur_bg', True, 0.5)),
        (make_state(blur_bg=0.6, whiten_teeth=0.5),
         [{'role': 'user', 'content': '再淡一些'}, {'role': 'assistant', 'content': '你想减轻背景虚化还是牙齿美白？'}],
         '牙齿美白轻一点', upd('whiten_teeth', True, 0.4)),
        (make_state(slim_face=0.4, enlarge_eyes=0.7),
         [{'role': 'user', 'content': '太夸张了，收一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '大眼收一点', upd('enlarge_eyes', True, 0.6)),
        (make_state(slim_face=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '再明显点'}, {'role': 'assistant', 'content': '你想加强瘦脸还是背景虚化？'}],
         '背景虚化再加一点', upd('blur_bg', True, 0.6)),
        (make_state(enlarge_eyes=0.4, smooth_skin=0.6),
         [{'role': 'user', 'content': '再多一点'}, {'role': 'assistant', 'content': '你想加强大眼还是磨皮？'}],
         '大眼加一点', upd('enlarge_eyes', True, 0.5)),
        (make_state(slim_face=0.3, whiten_teeth=0.6),
         [{'role': 'user', 'content': '这个减弱一点'}, {'role': 'assistant', 'content': '你想减弱瘦脸还是牙齿美白？'}],
         '牙齿美白减弱一点', upd('whiten_teeth', True, 0.5)),
        (make_state(slim_face=0.7, enlarge_eyes=0.4, smooth_skin=0.5),
         [{'role': 'user', 'content': '自然一点'}, {'role': 'assistant', 'content': '你想调轻哪一项效果？'}],
         '瘦脸减轻一点', upd('slim_face', True, 0.6)),
        (make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '把大眼调小一点', upd('enlarge_eyes', True, 0.4)),
        (make_state(slim_face=0.5, remove_acne=0.6),
         [{'role': 'user', 'content': '这个去掉'}, {'role': 'assistant', 'content': '你想关闭瘦脸还是祛痘？'}],
         '把祛痘关掉', upd('remove_acne', False, 0.6)),
        (make_state(blur_bg=0.5, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '关一个'}, {'role': 'assistant', 'content': '你想关闭背景虚化还是大眼？'}],
         '关闭大眼', upd('enlarge_eyes', False, 0.0)),
        (make_state(smooth_skin=0.5, whiten_teeth=0.5),
         [{'role': 'user', 'content': '这个不要了'}, {'role': 'assistant', 'content': '你想关闭磨皮还是牙齿美白？'}],
         '关掉牙齿美白', upd('whiten_teeth', False, 0.0)),
        (make_state(slim_face=0.5, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '再淡一点'}, {'role': 'assistant', 'content': '你想调淡瘦脸、大眼还是背景虚化？'}],
         '背景虚化再淡一点', upd('blur_bg', True, 0.4)),
        (make_state(slim_face=0.4, smooth_skin=0.7),
         [{'role': 'user', 'content': '退一格'}, {'role': 'assistant', 'content': '你想调低瘦脸还是磨皮？'},
          {'role': 'user', 'content': '磨皮'}, {'role': 'assistant', 'content': '你想调低瘦脸还是磨皮？'}],
         '磨皮降低一点', upd('smooth_skin', True, 0.6)),
        (make_state(enlarge_eyes=0.6, whiten_teeth=0.4),
         [{'role': 'user', 'content': '再强一点'}, {'role': 'assistant', 'content': '你想加强大眼还是牙齿美白？'}],
         '牙齿再白一点', upd('whiten_teeth', True, 0.5)),
        (make_state(remove_acne=0.3, blur_bg=0.5),
         [{'role': 'user', 'content': '再加重一点'}, {'role': 'assistant', 'content': '你想加强祛痘还是背景虚化？'}],
         '祛痘敏感度提高一点', upd('remove_acne', True, 0.4)),
        (make_state(slim_face=0.6, enlarge_eyes=0.6),
         [{'role': 'user', 'content': '减一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '只调大眼，减轻一点', upd('enlarge_eyes', True, 0.5)),
        (make_state(slim_face=0.6, enlarge_eyes=0.6),
         [{'role': 'user', 'content': '减一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '只把瘦脸减一点', upd('slim_face', True, 0.5)),
    ]
    for st, hist, instr, plan in a1_cases:
        add('A1_clarify_recover_target', st, hist, instr, plan)

    # A2 (20 rows): After 1 or 2 clarify turns, user gives explicit numeric value -> update
    a2_cases = [
        (make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '眼睛小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'},
          {'role': 'user', 'content': '调整大眼，眼睛小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '眼睛调到0.2', upd('enlarge_eyes', True, 0.2)),
        (make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '再小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '大眼强度设为 0.3', upd('enlarge_eyes', True, 0.3)),
        (make_state(slim_face=0.5, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '瘦脸调到 0.2', upd('slim_face', True, 0.2)),
        (make_state(slim_face=0.6, smooth_skin=0.5),
         [{'role': 'user', 'content': '再淡点'}, {'role': 'assistant', 'content': '你想调整瘦脸还是磨皮？'}],
         '磨皮改成 0.3', upd('smooth_skin', True, 0.3)),
        (make_state(blur_bg=0.5, whiten_teeth=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想减轻背景虚化还是牙齿美白？'}],
         '背景虚化调到 0.2', upd('blur_bg', True, 0.2)),
        (make_state(blur_bg=0.5, whiten_teeth=0.5),
         [{'role': 'user', 'content': '调淡点'}, {'role': 'assistant', 'content': '你想减轻背景虚化还是牙齿美白？'}],
         '牙齿美白设为 0.3', upd('whiten_teeth', True, 0.3)),
        (make_state(remove_acne=0.5, smooth_skin=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整祛痘还是磨皮？'}],
         '祛痘敏感度设为 0.2', upd('remove_acne', True, 0.2)),
        (make_state(slim_face=0.5, enlarge_eyes=0.6, smooth_skin=0.4),
         [{'role': 'user', 'content': '改一下'}, {'role': 'assistant', 'content': '请明确要调整哪一项？'}],
         '大眼改成 0.2', upd('enlarge_eyes', True, 0.2)),
        (make_state(slim_face=0.5, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '眼睛强度调到 0.4', upd('enlarge_eyes', True, 0.4)),
        (make_state(slim_face=0.5, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '脸部瘦脸设为 0.3', upd('slim_face', True, 0.3)),
        (make_state(blur_bg=0.7, slim_face=0.4),
         [{'role': 'user', 'content': '再淡一点'}, {'role': 'assistant', 'content': '你想调整背景虚化还是瘦脸？'}],
         '背景调到 0.4', upd('blur_bg', True, 0.4)),
        (make_state(whiten_teeth=0.7, enlarge_eyes=0.4),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整牙齿美白还是大眼？'}],
         '牙白调到 0.4', upd('whiten_teeth', True, 0.4)),
        (make_state(remove_acne=0.7, slim_face=0.4),
         [{'role': 'user', 'content': '再低一点'}, {'role': 'assistant', 'content': '你想调整祛痘还是瘦脸？'}],
         '祛痘调到 0.3', upd('remove_acne', True, 0.3)),
        (make_state(smooth_skin=0.7, blur_bg=0.5),
         [{'role': 'user', 'content': '减一点'}, {'role': 'assistant', 'content': '你想减轻磨皮还是背景虚化？'}],
         '磨皮设为 0.4', upd('smooth_skin', True, 0.4)),
        (make_state(slim_face=0.4, enlarge_eyes=0.6, blur_bg=0.5),
         [{'role': 'user', 'content': '再小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'},
          {'role': 'user', 'content': '小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '把大眼调到 0.1', upd('enlarge_eyes', True, 0.1)),
        (make_state(slim_face=0.6, enlarge_eyes=0.4),
         [{'role': 'user', 'content': '不够自然'}, {'role': 'assistant', 'content': '你想调整哪一项效果？'}],
         '瘦脸改到 0.3', upd('slim_face', True, 0.3)),
        (make_state(enlarge_eyes=0.5, blur_bg=0.6),
         [{'role': 'user', 'content': '降一下'}, {'role': 'assistant', 'content': '你想调整大眼还是背景虚化？'}],
         '眼睛调到 0.3', upd('enlarge_eyes', True, 0.3)),
        (make_state(smooth_skin=0.5, remove_acne=0.5),
         [{'role': 'user', 'content': '再强一点'}, {'role': 'assistant', 'content': '你想加强磨皮还是祛痘？'}],
         '磨皮强度调到 0.8', upd('smooth_skin', True, 0.8)),
        (make_state(whiten_teeth=0.5, slim_face=0.5),
         [{'role': 'user', 'content': '再高一点'}, {'role': 'assistant', 'content': '你想加强牙齿美白还是瘦脸？'}],
         '牙齿美白调到 0.8', upd('whiten_teeth', True, 0.8)),
        (make_state(enlarge_eyes=0.7, smooth_skin=0.5),
         [{'role': 'user', 'content': '调整一下'}, {'role': 'assistant', 'content': '你想调整大眼还是磨皮？'}],
         '大眼设为 0.4', upd('enlarge_eyes', True, 0.4)),
    ]
    for st, hist, instr, plan in a2_cases:
        add('A2_clarify_recover_numeric', st, hist, instr, plan)

    # A3 (12 rows): After clarify, user pivots to new tool / undo / reset / unsupported
    a3_cases = [
        (make_state(slim_face=0.4, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '先帮我虚化背景', upd('blur_bg', True, 0.5)),
        (make_state(slim_face=0.4, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '帮我美白牙齿', upd('whiten_teeth', True, 0.5)),
        (make_state(slim_face=0.4, blur_bg=0.5),
         [{'role': 'user', 'content': '淡一点'}, {'role': 'assistant', 'content': '你想调整瘦脸还是背景虚化？'}],
         '给我磨一下皮', upd('smooth_skin', True, 0.5)),
        (make_state(enlarge_eyes=0.5, smooth_skin=0.5),
         [{'role': 'user', 'content': '减一点'}, {'role': 'assistant', 'content': '你想调整大眼还是磨皮？'}],
         '帮我祛痘', upd('remove_acne', True, 0.5)),
        (make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '算了，把大眼关掉', upd('enlarge_eyes', False, 0.0)),
        (make_state(slim_face=0.4, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '撤销上一步', {'action': 'undo', 'updates': [], 'question': None}),
        (make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '恢复原图', {'action': 'reset', 'updates': [], 'question': None}),
        (make_state(smooth_skin=0.5, remove_acne=0.5),
         [{'role': 'user', 'content': '都去掉'}, {'role': 'assistant', 'content': '你想关闭磨皮还是祛痘？'}],
         '撤销刚才那一步', {'action': 'undo', 'updates': [], 'question': None}),
        (make_state(slim_face=0.5, whiten_teeth=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整瘦脸还是牙齿美白？'}],
         '全部重置', {'action': 'reset', 'updates': [], 'question': None}),
        (make_state(slim_face=0.5, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '不用管刚才的，给背景加点虚化到 0.4', upd('blur_bg', True, 0.4)),
        (make_state(blur_bg=0.5, smooth_skin=0.5),
         [{'role': 'user', 'content': '淡一点'}, {'role': 'assistant', 'content': '你想调整背景虚化还是磨皮？'}],
         '瘦脸强度调到 0.3', upd('slim_face', True, 0.3)),
        (make_state(remove_acne=0.5),
         [{'role': 'user', 'content': '都去掉'}, {'role': 'assistant', 'content': '你想把痘痘全部去除还是关闭祛痘？'}],
         '放大眼睛和磨皮', {'action': 'update', 'updates': [
             {'tool': 'enlarge_eyes', 'enabled': True, 'strength': 0.5},
             {'tool': 'smooth_skin', 'enabled': True, 'strength': 0.5},
         ], 'question': None}),
    ]
    for st, hist, instr, plan in a3_cases:
        add('A3_clarify_pivot_or_control', st, hist, instr, plan)

    # ------------------------------------------------------------------
    # Bucket B (48 rows): Oral relative decrease/increase under multi-tool active states
    # Fixes the 0/1800 "眼睛小一点" -> update blind spot!
    # ------------------------------------------------------------------
    b_cases = [
        # Enlarge eyes oral decrease (16 rows)
        (make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5), [], '眼睛小一点', upd('enlarge_eyes', True, 0.4)),
        (make_state(slim_face=0.5, enlarge_eyes=0.6), [], '眼睛再小一点', upd('enlarge_eyes', True, 0.5)),
        (make_state(slim_face=0.4, enlarge_eyes=0.5, smooth_skin=0.5), [], '眼睛太大了，修小一点', upd('enlarge_eyes', True, 0.4)),
        (make_state(slim_face=0.3, enlarge_eyes=0.7), [], '眼睛别放那么大，收一点', upd('enlarge_eyes', True, 0.6)),
        (make_state(enlarge_eyes=0.5, blur_bg=0.5), [], '大眼小一点', upd('enlarge_eyes', True, 0.4)),
        (make_state(enlarge_eyes=0.4, remove_acne=0.5, smooth_skin=0.5), [], '把眼睛修小一点', upd('enlarge_eyes', True, 0.3)),
        (make_state(enlarge_eyes=0.6, whiten_teeth=0.5), [], '大眼效果调小一格', upd('enlarge_eyes', True, 0.5)),
        (make_state(enlarge_eyes=0.8, slim_face=0.5), [], '眼睛放得太夸张了，减轻一点', upd('enlarge_eyes', True, 0.7)),
        (make_state(enlarge_eyes=0.3, slim_face=0.4), [], '眼睛再收小些', upd('enlarge_eyes', True, 0.2)),
        (make_state(enlarge_eyes=0.5, smooth_skin=0.4), [], '大眼淡一点', upd('enlarge_eyes', True, 0.4)),
        (make_state(enlarge_eyes=0.5, slim_face=0.5), [], '大眼轻一点，瘦脸别动', upd('enlarge_eyes', True, 0.4)),
        (make_state(enlarge_eyes=0.6, blur_bg=0.4), [], '眼睛稍微缩回一点', upd('enlarge_eyes', True, 0.5)),
        (make_state(enlarge_eyes=0.5, slim_face=0.4, blur_bg=0.5), [], '眼睛调到 0.2', upd('enlarge_eyes', True, 0.2)),
        (make_state(enlarge_eyes=0.7, slim_face=0.5), [], '眼睛改小到 0.3', upd('enlarge_eyes', True, 0.3)),
        (make_state(enlarge_eyes=0.4, slim_face=0.5), [], '眼睛再大一点', upd('enlarge_eyes', True, 0.5)),
        (make_state(enlarge_eyes=0.5, blur_bg=0.5), [], '大眼再加一点', upd('enlarge_eyes', True, 0.6)),
        # Slim face oral increase/decrease (10 rows)
        (make_state(slim_face=0.4, enlarge_eyes=0.5), [], '脸还是宽，再收一点', upd('slim_face', True, 0.5)),
        (make_state(slim_face=0.5, blur_bg=0.5), [], '脸颊还是显宽，再往里收些', upd('slim_face', True, 0.6)),
        (make_state(slim_face=0.6, enlarge_eyes=0.5), [], '脸瘦得太尖了，放宽一点', upd('slim_face', True, 0.5)),
        (make_state(slim_face=0.7, smooth_skin=0.5), [], '瘦脸别太狠，轻一点', upd('slim_face', True, 0.6)),
        (make_state(slim_face=0.5, enlarge_eyes=0.4, blur_bg=0.5), [], '脸再小一点', upd('slim_face', True, 0.6)),
        (make_state(slim_face=0.4, whiten_teeth=0.5), [], '瘦脸再加一格', upd('slim_face', True, 0.5)),
        (make_state(slim_face=0.5, remove_acne=0.5), [], '瘦脸减轻一点', upd('slim_face', True, 0.4)),
        (make_state(slim_face=0.8, enlarge_eyes=0.5), [], '瘦脸收回来一点', upd('slim_face', True, 0.7)),
        (make_state(slim_face=0.3, blur_bg=0.6), [], '瘦脸再突出一点', upd('slim_face', True, 0.4)),
        (make_state(slim_face=0.5, smooth_skin=0.5), [], '瘦脸调到 0.2，磨皮不变', upd('slim_face', True, 0.2)),
        # Smooth skin, blur_bg, whiten_teeth, remove_acne oral tweaks (22 rows)
        (make_state(smooth_skin=0.6, slim_face=0.4), [], '皮肤纹理多留一点', upd('smooth_skin', True, 0.5)),
        (make_state(smooth_skin=0.7, enlarge_eyes=0.5), [], '磨皮太假了，轻一点', upd('smooth_skin', True, 0.6)),
        (make_state(smooth_skin=0.4, remove_acne=0.5), [], '皮肤再磨细一点', upd('smooth_skin', True, 0.5)),
        (make_state(smooth_skin=0.5, blur_bg=0.5), [], '磨皮淡一点，背景不动', upd('smooth_skin', True, 0.4)),
        (make_state(smooth_skin=0.3, whiten_teeth=0.5), [], '磨皮再加一点', upd('smooth_skin', True, 0.4)),
        (make_state(blur_bg=0.6, slim_face=0.4), [], '背景不要那么糊，淡一点', upd('blur_bg', True, 0.5)),
        (make_state(blur_bg=0.7, enlarge_eyes=0.5), [], '背景虚化减轻些', upd('blur_bg', True, 0.6)),
        (make_state(blur_bg=0.5, smooth_skin=0.5), [], '后景虚化往下调一格', upd('blur_bg', True, 0.4)),
        (make_state(blur_bg=0.4, slim_face=0.4, enlarge_eyes=0.5), [], '背景再虚化一点', upd('blur_bg', True, 0.5)),
        (make_state(blur_bg=0.8, whiten_teeth=0.5), [], '背景虚化收轻一点', upd('blur_bg', True, 0.7)),
        (make_state(blur_bg=0.3, remove_acne=0.5), [], '背景模糊加强一点', upd('blur_bg', True, 0.4)),
        (make_state(whiten_teeth=0.7, slim_face=0.4), [], '牙齿别太白，轻一点', upd('whiten_teeth', True, 0.6)),
        (make_state(whiten_teeth=0.6, blur_bg=0.5), [], '牙色自然些，调淡一点', upd('whiten_teeth', True, 0.5)),
        (make_state(whiten_teeth=0.5, enlarge_eyes=0.5), [], '牙齿美白减弱一点', upd('whiten_teeth', True, 0.4)),
        (make_state(whiten_teeth=0.4, smooth_skin=0.5), [], '牙齿再白一点', upd('whiten_teeth', True, 0.5)),
        (make_state(whiten_teeth=0.8, blur_bg=0.4), [], '牙齿美白往下调一格', upd('whiten_teeth', True, 0.7)),
        (make_state(whiten_teeth=0.3, slim_face=0.5), [], '牙齿提亮再加一点', upd('whiten_teeth', True, 0.4)),
        (make_state(remove_acne=0.6, smooth_skin=0.5), [], '祛痘敏感度降一点', upd('remove_acne', True, 0.5)),
        (make_state(remove_acne=0.4, slim_face=0.4), [], '祛痘再加强一点', upd('remove_acne', True, 0.5)),
        (make_state(remove_acne=0.8, enlarge_eyes=0.5), [], '祛痘轻一点，别修太重', upd('remove_acne', True, 0.7)),
        (make_state(remove_acne=0.5, blur_bg=0.5), [], '祛痘敏感度往下调一格', upd('remove_acne', True, 0.4)),
        (make_state(remove_acne=0.3, whiten_teeth=0.5), [], '祛痘敏感度提高一点', upd('remove_acne', True, 0.4)),
    ]
    for st, hist, instr, plan in b_cases:
        add('B_oral_relative_multi_tool', st, hist, instr, plan)

    # ------------------------------------------------------------------
    # Bucket C (40 rows): 5 strength tools paired non-zero (-0.1) vs 0.1->zero confirm
    # Fixes whiten_teeth 0 ordinary-decrease rows & final_v1_09 (0.7 falsely asking zero!)
    # ------------------------------------------------------------------
    tool_cn = {
        'slim_face': ('瘦脸', '瘦脸稍微退一格', '瘦脸往下调一档'),
        'enlarge_eyes': ('大眼', '大眼再轻一点', '大眼往下调一档'),
        'smooth_skin': ('磨皮', '磨皮再淡一点', '磨皮往下调一档'),
        'blur_bg': ('背景虚化', '背景虚化再淡一点', '背景虚化再调低一档'),
        'whiten_teeth': ('牙齿美白', '牙齿美白再轻一点', '牙齿美白往下调一档'),
    }
    for tool, (cn, phrase_a, phrase_b) in tool_cn.items():
        # 5 non-zero decrease levels -> MUST update, NEVER ask confirm_zero!
        add('C_nonzero_decrease', make_state(**{tool: 0.2}), [], phrase_a, upd(tool, True, 0.1))
        add('C_nonzero_decrease', make_state(**{tool: 0.4}), [], phrase_b, upd(tool, True, 0.3))
        add('C_nonzero_decrease', make_state(**{tool: 0.5}), [], '稍微减轻一些', upd(tool, True, 0.4))
        add('C_nonzero_decrease', make_state(**{tool: 0.7}), [], '再收轻一点', upd(tool, True, 0.6))
        add('C_nonzero_decrease', make_state(**{tool: 0.9}), [], phrase_a, upd(tool, True, 0.8))
        # 3 true 0.1 zero-boundary cases (ask, confirm, cancel)
        q_zero = f'再降低一点会让{cn}强度变成 0，效果将消失。要关闭{cn}吗？'
        add('C_zero_boundary', make_state(**{tool: 0.1}), [], phrase_b, clr(q_zero))
        add('C_zero_boundary', make_state(**{tool: 0.1}),
            [{'role': 'user', 'content': phrase_a}, {'role': 'assistant', 'content': q_zero}],
            '确认关闭', upd(tool, False, 0.0))
        add('C_zero_boundary', make_state(**{tool: 0.1}),
            [{'role': 'user', 'content': phrase_a}, {'role': 'assistant', 'content': q_zero}],
            '不关，先保留', clr('好的，保持当前设置。还想调整哪一项？'))

    # ------------------------------------------------------------------
    # Bucket D (24 rows): Explicit disable (no extra clarify), remove_acne=0, & true ambiguous clarify
    # Fixes final_v1_18 (explicit disable falsely clarifying) while keeping real ambiguity safe!
    # ------------------------------------------------------------------
    d_cases = [
        # 12 explicit disables -> MUST update enabled=False, NEVER clarify
        (make_state(enlarge_eyes=0.6), [], '把大眼关掉', upd('enlarge_eyes', False, 0.0)),
        (make_state(enlarge_eyes=0.5, slim_face=0.4), [], '大眼不要了', upd('enlarge_eyes', False, 0.0)),
        (make_state(slim_face=0.5, blur_bg=0.5), [], '关闭瘦脸', upd('slim_face', False, 0.0)),
        (make_state(slim_face=0.3), [], '把瘦脸取消掉', upd('slim_face', False, 0.0)),
        (make_state(smooth_skin=0.5, remove_acne=0.5), [], '磨皮关掉', upd('smooth_skin', False, 0.0)),
        (make_state(smooth_skin=0.7), [], '不要磨皮了', upd('smooth_skin', False, 0.0)),
        (make_state(blur_bg=0.5, whiten_teeth=0.5), [], '把背景虚化关掉', upd('blur_bg', False, 0.0)),
        (make_state(blur_bg=0.4), [], '取消背景虚化', upd('blur_bg', False, 0.0)),
        (make_state(whiten_teeth=0.7), [], '关掉牙齿美白', upd('whiten_teeth', False, 0.0)),
        (make_state(whiten_teeth=0.5, slim_face=0.4), [], '牙齿美白不用开了', upd('whiten_teeth', False, 0.0)),
        (make_state(enlarge_eyes=0.5, slim_face=0.4, blur_bg=0.5), [], '只保留瘦脸，其余都关掉',
         {'action': 'update', 'updates': [
             {'tool': 'enlarge_eyes', 'enabled': False, 'strength': 0.0},
             {'tool': 'blur_bg', 'enabled': False, 'strength': 0.0},
         ], 'question': None}),
        (make_state(smooth_skin=0.5, remove_acne=0.5, whiten_teeth=0.4), [], '只留磨皮，其他都停用',
         {'action': 'update', 'updates': [
             {'tool': 'remove_acne', 'enabled': False, 'sensitivity': 0.5},
             {'tool': 'whiten_teeth', 'enabled': False, 'strength': 0.0},
         ], 'question': None}),
        # 6 remove_acne sensitivity=0 valid & disable preserves sensitivity
        (make_state(remove_acne=0.1), [], '祛痘敏感度再往下调一点', upd('remove_acne', True, 0.0)),
        (make_state(remove_acne=0.1, smooth_skin=0.5), [], '祛痘再降一格', upd('remove_acne', True, 0.0)),
        (make_state(remove_acne=0.2), [], '祛痘敏感度设为 0', upd('remove_acne', True, 0.0)),
        (make_state(remove_acne=0.5), [], '把祛痘关掉', upd('remove_acne', False, 0.5)),
        (make_state(remove_acne=0.3, slim_face=0.5), [], '关闭祛痘，瘦脸不变', upd('remove_acne', False, 0.3)),
        (make_state(remove_acne=0.0), [], '祛痘也关掉', upd('remove_acne', False, 0.0)),
        # 6 genuine multi-tool ambiguous requests -> MUST keep clarifying
        (make_state(slim_face=0.4, enlarge_eyes=0.5), [], '再轻一点', clr('当前瘦脸和大眼都已开启，你想减轻哪一项？')),
        (make_state(blur_bg=0.5, whiten_teeth=0.5), [], '再淡一点', clr('当前背景虚化和牙齿美白都已开启，你想调淡哪一项？')),
        (make_state(smooth_skin=0.5, remove_acne=0.5), [], '这个去掉', clr('当前磨皮和祛痘都已开启，你想关闭哪一项？')),
        (make_state(slim_face=0.5, smooth_skin=0.5, blur_bg=0.5), [], '再加一点', clr('当前多项效果已开启，你想加强哪一项？')),
        (make_state(enlarge_eyes=0.5, blur_bg=0.5), [], '帮我把背景换成海边', clr('目前不支持换背景；你想改用可用的瘦脸、大眼、祛痘、磨皮、背景虚化或牙齿美白吗？')),
        (make_state(slim_face=0.5), [], '把眼镜擦掉', clr('目前不支持移除眼镜；你想改用可用的瘦脸、大眼、祛痘、磨皮、背景虚化或牙齿美白吗？')),
    ]
    for st, hist, instr, plan in d_cases:
        add('D_disable_acne_and_ambiguity', st, hist, instr, plan)

    assert len(specs) == 168, f'Expected 168 targeted specs, got {len(specs)}'
    return specs


def build_regression_cases(dev_rows: list[dict]) -> list[dict]:
    """Build 12 diagnostic regression cases (using dev images, separate from dev280)."""
    # Pick 12 distinct dev images for realistic VLM inference
    seen = set()
    imgs = []
    for r in dev_rows:
        if r['image_id'] not in seen:
            seen.add(r['image_id'])
            imgs.append(r)
        if len(imgs) == 12:
            break

    cases = [
        # Exact reproduction of the user's live screenshot failure!
        ('reg_01_eye_small_multi',
         make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5), [],
         '眼睛小一点', upd('enlarge_eyes', True, 0.4)),
        ('reg_02_clarify_recover_eye',
         make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '眼睛小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '调整大眼，眼睛小一点', upd('enlarge_eyes', True, 0.4)),
        ('reg_03_double_clarify_numeric',
         make_state(slim_face=0.4, enlarge_eyes=0.5, blur_bg=0.5),
         [{'role': 'user', 'content': '眼睛小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'},
          {'role': 'user', 'content': '调整大眼，眼睛小一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '眼睛调到0.2', upd('enlarge_eyes', True, 0.2)),
        # Reproduction of final_v1_09 (teeth 0.7 -> 0.6, NOT confirm_zero!)
        ('reg_04_teeth_07_decrease',
         make_state(whiten_teeth=0.7), [],
         '再轻一点', upd('whiten_teeth', True, 0.6)),
        # Reproduction of final_v1_18 (explicit disable enlarge_eyes, NOT clarify!)
        ('reg_05_disable_eyes_explicit',
         make_state(enlarge_eyes=0.6), [],
         '把大眼关掉', upd('enlarge_eyes', False, 0.0)),
        # Reproduction of final_v1_20 (clarify then specify smooth_skin)
        ('reg_06_clarify_recover_smooth',
         make_state(smooth_skin=0.5, remove_acne=0.5),
         [{'role': 'user', 'content': '再轻一点'}, {'role': 'assistant', 'content': '你想调整磨皮还是祛痘？'}],
         '我说的是磨皮，再降一点', upd('smooth_skin', True, 0.4)),
        # True 0.1 zero confirm still works
        ('reg_07_teeth_01_zero_confirm',
         make_state(whiten_teeth=0.1), [],
         '牙齿美白再淡一点', clr('再降低一点会让牙齿美白强度变成 0，效果将消失。要关闭牙齿美白吗？')),
        # Acne 0.1 -> 0.0 stays enabled
        ('reg_08_acne_01_to_00',
         make_state(remove_acne=0.1), [],
         '祛痘敏感度再降一点', upd('remove_acne', True, 0.0)),
        # Slim face oral increase ("脸还是宽再收一点")
        ('reg_09_face_wide_increase',
         make_state(slim_face=0.4, enlarge_eyes=0.5), [],
         '感觉脸还是宽，再收一点', upd('slim_face', True, 0.5)),
        # Pivot from clarify to undo
        ('reg_10_clarify_pivot_undo',
         make_state(slim_face=0.4, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '撤销上一步', {'action': 'undo', 'updates': [], 'question': None}),
        # Pivot from clarify to new tool
        ('reg_11_clarify_pivot_new_tool',
         make_state(slim_face=0.4, enlarge_eyes=0.5),
         [{'role': 'user', 'content': '轻一点'}, {'role': 'assistant', 'content': '你想调整大眼还是瘦脸？'}],
         '帮我虚化背景', upd('blur_bg', True, 0.5)),
        # Genuine ambiguity still clarifies
        ('reg_12_genuine_ambiguity',
         make_state(slim_face=0.5, enlarge_eyes=0.5), [],
         '再轻一点', clr('当前瘦脸和大眼都已开启，你想减轻哪一项？')),
    ]
    out = []
    for (cid, st, hist, instr, plan), img_row in zip(cases, imgs):
        validate_plan_structure(plan)
        out.append({
            'sample_id': cid,
            'image_id': img_row['image_id'],
            'image_ref': img_row['image_ref'],
            'current_state': st,
            'history': hist,
            'instruction': instr,
            'expected_plan': plan,
        })
    return out


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    prompt_text = V1_PROMPT.read_text(encoding='utf-8')
    OUT_PROMPT.write_text(prompt_text, encoding='utf-8')

    base_rows = [json.loads(line) for line in V1_TRAIN_DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
    dev_rows = [json.loads(line) for line in DEV280.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(base_rows) == 1800 and len(dev_rows) == 280

    # Collect unique train image metadata from the frozen 1800 set
    train_images: list[dict] = []
    seen_imgs: set[str] = set()
    for row in base_rows:
        if row['image_id'] not in seen_imgs:
            seen_imgs.add(row['image_id'])
            train_images.append({
                'image_id': row['image_id'],
                'image_ref': row['image_ref'],
                'image_sha256': row['image_sha256'],
                'subject_group': row['subject_group'],
                'abs_path': row['messages'][0]['content'][0]['image'],
            })
    assert len(train_images) == 199

    # Strict anti-leakage sets from dev280
    dev_image_ids = {r['image_id'] for r in dev_rows}
    dev_image_shas = {r.get('image_sha256') for r in dev_rows if r.get('image_sha256')}
    dev_groups = {r.get('subject_group') for r in dev_rows if r.get('subject_group')}
    dev_pairs = {
        (json.dumps(r['current_state'], sort_keys=True), r['instruction'].strip().rstrip('。'))
        for r in dev_rows
    }

    for img in train_images:
        assert img['image_id'] not in dev_image_ids
        assert img['image_sha256'] not in dev_image_shas
        assert img['subject_group'] not in dev_groups
        assert Path(img['abs_path']).is_file()

    rng = random.Random(20261126)
    shuffled_imgs = list(train_images)
    rng.shuffle(shuffled_imgs)

    specs = build_targeted_specs()
    new_rows: list[dict] = []
    bucket_counts: dict[str, int] = {}

    for idx, spec in enumerate(specs, 1):
        img = shuffled_imgs[(idx - 1) % len(shuffled_imgs)]
        pair_key = (json.dumps(spec['state'], sort_keys=True), spec['instruction'].strip().rstrip('。'))
        if pair_key in dev_pairs:
            raise SystemExit(f'Leakage detected against dev280: {spec["instruction"]}')
        text = (
            prompt_text
            .replace('{state}', json.dumps(spec['state'], ensure_ascii=False))
            .replace('{history}', json.dumps(spec['history'], ensure_ascii=False))
            .replace('{instruction}', spec['instruction'])
        )
        completion = json.dumps(spec['plan'], ensure_ascii=False, separators=(',', ':'))
        row = {
            'sample_id': f'six_v2_targeted_{idx:03d}',
            'image_id': img['image_id'],
            'image_ref': img['image_ref'],
            'image_sha256': img['image_sha256'],
            'subject_group': img['subject_group'],
            'project_split': 'train',
            'annotation_source': 'targeted_contrast_v2',
            'review_status': 'assistant_authored_pending_user_spot_check',
            'provenance_label': 'targeted_contrast_v2_' + spec['bucket'],
            'source_eligible_for_training': True,
            'eligible_for_training': True,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img['abs_path']},
                    {'type': 'text', 'text': text},
                ],
            }],
            'completion': completion,
            'qualification_scope': 'six_tool_plan_only_sft_v2',
            'targeted_bucket': spec['bucket'],
        }
        new_rows.append(row)
        bucket_counts[spec['bucket']] = bucket_counts.get(spec['bucket'], 0) + 1

    # Write targeted 168 rows and merged 1968 rows (keeping original 1800 intact, updating scope only in v2 copy)
    with OUT_TARGETED.open('w', encoding='utf-8') as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    merged = []
    for r in base_rows:
        cp = copy.deepcopy(r)
        cp['qualification_scope'] = 'six_tool_plan_only_sft_v2'
        merged.append(cp)
    merged.extend(new_rows)
    assert len(merged) == 1968
    assert len({r['sample_id'] for r in merged}) == 1968

    with OUT_MERGED.open('w', encoding='utf-8') as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Write regression suite
    reg_cases = build_regression_cases(dev_rows)
    with OUT_REGRESSION.open('w', encoding='utf-8') as f:
        for r in reg_cases:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # CPU processor token-boundary verification on all 168 new rows (ensures zero runtime surprises!)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(ROOT / 'agent_baseline/models/Qwen3-VL-2B-Instruct'), local_files_only=True)
    max_new_tokens = 0
    for r in new_rows:
        prefix = processor.apply_chat_template(
            r['messages'], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt'
        )
        full = processor.apply_chat_template(
            r['messages'] + [{'role': 'assistant', 'content': r['completion']}],
            tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt'
        )
        pids, fids = prefix['input_ids'][0], full['input_ids'][0]
        if len(pids) >= len(fids) or len(fids) > 3072 or not (pids == fids[:len(pids)]).all().item():
            raise SystemExit(f"Processor boundary check failed: {r['sample_id']} len={len(fids)}")
        max_new_tokens = max(max_new_tokens, int(len(fids)))

    # Write human-readable review summary markdown
    md_lines = [
        '# 六工具定向补训样本抽查清单（168 条，v2）',
        '',
        '- 原始训练集保持冻结不动：`train_plan_only_1800.jsonl`（1800 条、199 图）',
        '- 本轮新增定向样本：`targeted_add_168.jsonl`（168 条，全部绑定原 199 张 train 图，与 dev280 零图片/零分组/零 `(state, instruction)` 重合）',
        '- 合并训练集：`train_plan_only_1968.jsonl`（1968 条，492 优化步，1 epoch）',
        f'- CPU Tokenizer 边界与长度校验：168/168 通过，最大 token 长度 `{max_new_tokens}` ≤ `3072`',
        '',
        '## 分类统计',
        '',
        '| 类别 | 条数 |',
        '| --- | ---: |',
    ]
    for k, v in sorted(bucket_counts.items()):
        md_lines.append(f'| `{k}` | {v} |')
    md_lines += ['', '## 代表样本预览（每类前 2 条）', '']
    shown: dict[str, int] = {}
    for spec in specs:
        b = spec['bucket']
        if shown.get(b, 0) >= 2:
            continue
        shown[b] = shown.get(b, 0) + 1
        md_lines.append(
            f"- **[{b}]** 历史 `{json.dumps(spec['history'], ensure_ascii=False)}` ｜ 指令：`{spec['instruction']}` → `{json.dumps(spec['plan'], ensure_ascii=False)}`"
        )
    OUT_REVIEW_MD.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    manifest = {
        'status': 'qualified_for_six_tool_plan_only_sft_v2',
        'base_train_rows': 1800,
        'targeted_add_rows': len(new_rows),
        'total_train_rows': len(merged),
        'train_images': len(train_images),
        'dev_diagnostic_rows': len(dev_rows),
        'regression_rows': len(reg_cases),
        'bucket_counts': bucket_counts,
        'max_new_row_tokens': max_new_tokens,
        'max_length': 3072,
        'base_train_sha256': sha(V1_TRAIN_DATA),
        'targeted_add_sha256': sha(OUT_TARGETED),
        'frozen_train_sha256': sha(OUT_MERGED),
        'dev_diagnostic_sha256': sha(DEV280),
        'regression_sha256': sha(OUT_REGRESSION),
        'prompt_sha256': sha(OUT_PROMPT),
        'anti_leakage_checked': True,
        'test_labels_used': False,
        'limits': [
            '168 targeted rows are assistant-authored structured contrast pairs bound only to existing 199 train images.',
            'Dev280 remains a diagnostic dev set; strict selection gates from second_round_decision.md apply.',
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': 'built_v2_data',
        'total_train_rows': len(merged),
        'targeted_add_rows': len(new_rows),
        'max_new_row_tokens': max_new_tokens,
        'regression_rows': len(reg_cases),
        'frozen_train_sha256': manifest['frozen_train_sha256'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
