#!/usr/bin/env python3
"""Unit tests for the Pi-inspired unified portrait-agent core (`src/`)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import MockPlanner, PortraitAgent
from src.engine.geometry import apply_displacement, new_displacement_field
from src.engine.optics import blur_background, whiten_teeth
from src.engine.skin import heal_blemishes, smooth_skin, to_float, to_uint8
from src.session import TreeSessionStore
from src.tools import RetouchAdapter, ToolOutcome

CANDIDATES = ROOT / 'agent_baseline/final_acceptance_v1/visual_candidates/images'
SMILE_IMG = CANDIDATES / 'CelebAMask-HQ/307.jpg'
CLOSED_IMG = CANDIDATES / 'FFHQ/00244.png'


def _ensure_sample_images(tmp: Path) -> tuple[Path, Path]:
    """若本地未下载数据集图片，则生成轻量合成测试图，确保干净克隆后单测零依赖通过。"""
    if SMILE_IMG.is_file() and CLOSED_IMG.is_file():
        return SMILE_IMG, CLOSED_IMG
    from PIL import Image
    smile = tmp / 'smile_sample.jpg'
    closed = tmp / 'closed_sample.png'
    Image.new('RGB', (128, 128), (210, 175, 155)).save(smile, 'JPEG')
    Image.new('RGB', (128, 128), (195, 165, 150)).save(closed, 'PNG')
    return smile, closed


def test_tree_branching_and_non_destructive_undo() -> None:
    with tempfile.TemporaryDirectory(prefix='pi_tree_') as tmp:
        smile_img, _ = _ensure_sample_images(Path(tmp))
        store = TreeSessionStore(Path(tmp))
        agent = PortraitAgent(store=store, adapter=RetouchAdapter(mode='mock'), planner=MockPlanner())
        proj = agent.init_session(smile_img)
        sid = proj['session_id']

        # Turn 1: whiten teeth 0.5
        r1 = agent.run_turn(sid, '帮我美白牙齿')
        n1_id = r1['entry']['id']
        assert r1['events'] == [
            'agent_start', 'turn_start', 'plan_ready',
            'tool_execution_start', 'tool_execution_end', 'turn_end', 'agent_end'
        ]
        assert r1['projection']['state']['whiten_teeth'] == {'enabled': True, 'strength': 0.5}

        # Branch A (from n1): slim face 0.4
        r2a = agent.run_turn(sid, '瘦脸强度调到 0.4')
        n2a_id = r2a['entry']['id']
        assert r2a['projection']['state']['slim_face'] == {'enabled': True, 'strength': 0.4}
        assert r2a['projection']['state']['whiten_teeth'] == {'enabled': True, 'strength': 0.5}

        # Fork Branch B directly from n1: blur bg 0.7 (without losing Branch A!)
        r2b = agent.run_turn(sid, '背景虚化调到 0.7', parent_id=n1_id)
        n2b_id = r2b['entry']['id']
        assert r2b['entry']['parent_id'] == n1_id
        assert r2b['projection']['state']['blur_bg'] == {'enabled': True, 'strength': 0.7}
        assert r2b['projection']['state']['slim_face']['enabled'] is False
        assert r2b['projection']['state']['whiten_teeth'] == {'enabled': True, 'strength': 0.5}
        assert len(r2b['projection']['branch_leaves']) == 2
        assert r2b['projection']['total_nodes'] == 3

        # Switch back to Branch A: slim_face 0.4 restored intact!
        proj_a = agent.switch_branch(sid, n2a_id)
        assert proj_a['state']['slim_face'] == {'enabled': True, 'strength': 0.4}
        assert proj_a['state']['blur_bg']['enabled'] is False

        # Non-destructive undo on Branch A returns to teeth 0.5 while keeping all nodes in events.jsonl
        r_undo = agent.run_turn(sid, '撤销上一步')
        assert r_undo['projection']['state']['slim_face']['enabled'] is False
        assert r_undo['projection']['state']['whiten_teeth'] == {'enabled': True, 'strength': 0.5}
        assert r_undo['projection']['total_nodes'] == 4

        # Switch back to Branch B: still untouched!
        proj_b = agent.switch_branch(sid, n2b_id)
        assert proj_b['state']['blur_bg'] == {'enabled': True, 'strength': 0.7}


def test_signal_lifecycle_and_llm_projection() -> None:
    with tempfile.TemporaryDirectory(prefix='pi_sig_') as tmp:
        _, closed_img = _ensure_sample_images(Path(tmp))
        store = TreeSessionStore(Path(tmp))
        agent = PortraitAgent(store=store, adapter=RetouchAdapter(mode='mock'), planner=MockPlanner())
        sid = agent.init_session(closed_img)['session_id']

        # Turn 1: blur_bg 0.5
        agent.run_turn(sid, '帮我虚化背景')
        # Turn 2: teeth whitening on closed mouth -> skipped + opens clarify signal
        r2 = agent.run_turn(sid, '帮我美白牙齿')
        assert r2['entry']['status'] == 'clarify'
        assert r2['entry']['ui_details']['skipped'][0]['tool'] == 'whiten_teeth'
        assert r2['projection']['state']['blur_bg'] == {'enabled': True, 'strength': 0.5}
        assert r2['projection']['open_signal']['status'] == 'open'

        ctx_open = store.convert_to_llm(sid)
        assert ctx_open['open_signal'] is not None
        # Ensure UI details / timings are NOT leaked into LLM dialogue
        assert all(set(msg.keys()) == {'role', 'content'} for msg in ctx_open['dialogue'])

        # Turn 3: new valid edit consumes/closes the earlier signal
        r3 = agent.run_turn(sid, '瘦脸强度调到 0.3')
        assert r3['entry']['status'] == 'rendered'
        assert r3['projection']['open_signal'] is None
        assert store.convert_to_llm(sid)['open_signal'] is None


def test_hooks_barrier_and_idempotency() -> None:
    call_count = {'plans': 0}

    def counting_override(ctx, instr):
        call_count['plans'] += 1
        return {'action': 'update', 'updates': [{'tool': 'slim_face', 'enabled': True, 'strength': 0.5}], 'question': None}

    def block_high_slim(ctx):
        for u in ctx['plan'].get('updates', []):
            if u.get('tool') == 'slim_face' and u.get('strength', 0) > 0.8:
                return {'block': True, 'reason': 'blocked by safety hook'}
        return None

    def tag_after(ctx, outcome: ToolOutcome) -> ToolOutcome:
        return ToolOutcome(
            output=outcome.output,
            structured_output=outcome.structured_output,
            details={**outcome.details, 'audited_by_hook': True},
            is_error=outcome.is_error,
            terminate=outcome.terminate,
        )

    with tempfile.TemporaryDirectory(prefix='pi_hooks_') as tmp:
        smile_img, _ = _ensure_sample_images(Path(tmp))
        store = TreeSessionStore(Path(tmp))
        agent = PortraitAgent(
            store=store,
            adapter=RetouchAdapter(mode='mock'),
            planner=MockPlanner(override_fn=counting_override),
            before_tool_hooks=[block_high_slim],
            after_tool_hooks=[tag_after],
        )
        sid = agent.init_session(smile_img)['session_id']

        # Idempotent submission with same request_id executes planner only once
        first = agent.run_turn(sid, '瘦脸', request_id='req-idempotent-001')
        second = agent.run_turn(sid, '瘦脸', request_id='req-idempotent-001')
        assert call_count['plans'] == 1
        assert first['replayed'] is False and second['replayed'] is True
        assert first['entry']['id'] == second['entry']['id']
        assert first['entry']['ui_details']['audited_by_hook'] is True

        # Trigger before_tool_call hook block
        agent.planner = MockPlanner(
            override_fn=lambda ctx, instr: {
                'action': 'update',
                'updates': [{'tool': 'slim_face', 'enabled': True, 'strength': 0.95}],
                'question': None,
            }
        )
        blocked = agent.run_turn(sid, '极端瘦脸')
        assert blocked['entry']['status'] == 'rejected'
        assert blocked['entry']['is_error'] is True
        assert 'blocked by safety hook' in blocked['entry']['ui_details']['reason']
        # State stays unchanged at 0.5
        assert blocked['projection']['state']['slim_face'] == {'enabled': True, 'strength': 0.5}


def test_repeated_clarify_loop_folding_in_projection() -> None:
    """Verify Pi-style convert_to_llm folds stuck consecutive clarifies while keeping raw events.jsonl intact."""
    with tempfile.TemporaryDirectory(prefix='pi_fold_') as tmp:
        store = TreeSessionStore(Path(tmp))
        # Planner returns one normal update, then two identical clarify echoes
        plans = iter([
            {'action': 'update', 'updates': [{'tool': 'enlarge_eyes', 'enabled': True, 'strength': 0.5}], 'question': None},
            {'action': 'clarify', 'updates': [], 'question': '你想调整大眼还是瘦脸？'},
            {'action': 'clarify', 'updates': [], 'question': '你想调整大眼还是瘦脸？'},
        ])
        agent = PortraitAgent(
            store=store,
            adapter=RetouchAdapter(mode='mock'),
            planner=MockPlanner(override_fn=lambda ctx, instr: next(plans)),
        )
        smile_img, _ = _ensure_sample_images(Path(tmp))
        sid = agent.init_session(smile_img)['session_id']
        agent.run_turn(sid, '帮我大眼')
        agent.run_turn(sid, '眼睛小一点')

        # After 1 clarify, model context still sees the open clarify
        ctx1 = store.convert_to_llm(sid)
        assert ctx1['clarify_loop_folded'] is False
        assert ctx1['open_signal'] is not None
        assert len(ctx1['dialogue']) == 4

        # After 2 identical consecutive clarifies, projection folds the stuck echo out of LLM dialogue
        agent.run_turn(sid, '调整大眼，眼睛小一点')
        ctx2 = store.convert_to_llm(sid)
        assert ctx2['clarify_loop_folded'] is True
        assert ctx2['open_signal'] is None
        assert len(ctx2['dialogue']) == 2  # only clean rendered turn 1 remains!
        # Raw tree audit keeps all 3 turns completely intact
        assert len(store.load_entries(sid)) == 3


def test_native_engine_core_operators() -> None:
    """验证原生美颜引擎的几何、肤质与光学算子数值稳定性（纯 CPU，无外部数据依赖）。"""
    import numpy as np

    rng = np.random.default_rng(42)
    img = rng.integers(80, 210, size=(128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[24:104, 24:104] = 1.0

    # 1. 高低频磨皮与祛痘
    healed = heal_blemishes(img, np.array([[64.0, 64.0, 4.0]], dtype=np.float32), face_width=80.0)
    assert healed.shape == img.shape and healed.dtype == np.uint8
    smoothed = smooth_skin(to_float(healed), mask, face_width=80.0, strength=0.6)
    assert smoothed.shape == (128, 128, 3) and float(smoothed.min()) >= 0.0 and float(smoothed.max()) <= 1.0

    # 2. 背景虚化与牙齿美白
    bokeh = blur_background(smoothed, mask, strength=0.5)
    assert bokeh.shape == smoothed.shape
    mouth = np.zeros((128, 128), dtype=np.float32)
    mouth[70:85, 50:78] = 1.0
    whitened = whiten_teeth(bokeh, mouth, strength=0.5)
    assert whitened.shape == bokeh.shape

    # 3. 几何位移场与雅可比防撕裂
    field = new_displacement_field((128, 128))
    field[40:80, 40:80, 0] = 3.5
    warped, scale = apply_displacement(to_uint8(whitened), field)
    assert warped.shape == img.shape and 0.0 < scale <= 1.0


def main() -> None:
    test_tree_branching_and_non_destructive_undo()
    test_signal_lifecycle_and_llm_projection()
    test_hooks_barrier_and_idempotency()
    test_repeated_clarify_loop_folding_in_projection()
    test_native_engine_core_operators()
    print(json.dumps({
        'unit_tests': 'passed',
        'verified': [
            'tree_branching_and_non_destructive_undo',
            'signal_lifecycle_and_llm_projection',
            'hooks_barrier_and_idempotency',
            'repeated_clarify_loop_folding_in_projection',
            'native_engine_core_operators',
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
