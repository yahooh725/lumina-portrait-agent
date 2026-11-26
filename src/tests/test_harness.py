#!/usr/bin/env python3
"""Lumina 核心单元测试：覆盖树状分支会话、上下文投影、钩子幂等与原生修图工具。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import MockPlanner, PortraitAgent
from src.engine.geometry import apply_displacement, new_displacement_field
from src.engine.optics import blur_background, whiten_teeth
from src.engine.skin import heal_blemishes, smooth_skin, to_float, to_uint8
from src.session import TreeSessionStore
from src.tools import RetouchAdapter, ToolOutcome


def _make_test_images(tmp: Path) -> tuple[Path, Path]:
    smile = tmp / 'smile_sample.jpg'
    closed = tmp / 'closed_sample.png'
    Image.new('RGB', (128, 128), (210, 175, 155)).save(smile, 'JPEG')
    Image.new('RGB', (128, 128), (195, 165, 150)).save(closed, 'PNG')
    return smile, closed


def test_tree_branching_and_non_destructive_undo() -> None:
    with tempfile.TemporaryDirectory(prefix='lumina_tree_') as tmp:
        smile_img, _ = _make_test_images(Path(tmp))
        store = TreeSessionStore(Path(tmp))
        agent = PortraitAgent(store=store, adapter=RetouchAdapter(mode='mock'), planner=MockPlanner())
        sid = agent.init_session(smile_img)['session_id']

        r1 = agent.run_turn(sid, '帮我美白牙齿')
        n1_id = r1['entry']['id']
        assert r1['projection']['state']['whiten_teeth'] == {'enabled': True, 'strength': 0.5}

        # 方案 A：瘦脸 0.4
        r2a = agent.run_turn(sid, '瘦脸强度调到 0.4')
        n2a_id = r2a['entry']['id']
        assert r2a['projection']['state']['slim_face'] == {'enabled': True, 'strength': 0.4}

        # 从第 1 轮原地派生方案 B：背景虚化 0.7（不破坏方案 A）
        r2b = agent.run_turn(sid, '背景虚化调到 0.7', parent_id=n1_id)
        n2b_id = r2b['entry']['id']
        assert r2b['entry']['parent_id'] == n1_id
        assert r2b['projection']['state']['blur_bg'] == {'enabled': True, 'strength': 0.7}
        assert r2b['projection']['state']['slim_face']['enabled'] is False
        assert len(r2b['projection']['branch_leaves']) == 2

        # 切回方案 A 并验证无损撤销
        proj_a = agent.switch_branch(sid, n2a_id)
        assert proj_a['state']['slim_face'] == {'enabled': True, 'strength': 0.4}
        r_undo = agent.run_turn(sid, '撤销上一步')
        assert r_undo['projection']['state']['slim_face']['enabled'] is False
        assert r_undo['projection']['state']['whiten_teeth'] == {'enabled': True, 'strength': 0.5}

        # 方案 B 依然完整保留
        proj_b = agent.switch_branch(sid, n2b_id)
        assert proj_b['state']['blur_bg'] == {'enabled': True, 'strength': 0.7}


def test_teeth_gate_and_context_projection() -> None:
    with tempfile.TemporaryDirectory(prefix='lumina_sig_') as tmp:
        _, closed_img = _make_test_images(Path(tmp))
        store = TreeSessionStore(Path(tmp))
        agent = PortraitAgent(store=store, adapter=RetouchAdapter(mode='mock'), planner=MockPlanner())
        sid = agent.init_session(closed_img)['session_id']

        agent.run_turn(sid, '帮我虚化背景')
        r2 = agent.run_turn(sid, '帮我美白牙齿')
        assert r2['entry']['status'] == 'clarify'
        assert r2['entry']['ui_details']['skipped'][0]['tool'] == 'whiten_teeth'
        assert r2['projection']['state']['blur_bg'] == {'enabled': True, 'strength': 0.5}

        ctx = store.convert_to_llm(sid)
        assert ctx['open_signal'] is not None
        assert all(set(msg.keys()) == {'role', 'content'} for msg in ctx['dialogue'])

        r3 = agent.run_turn(sid, '瘦脸强度调到 0.3')
        assert r3['entry']['status'] == 'rendered'
        assert store.convert_to_llm(sid)['open_signal'] is None


def test_hooks_barrier_and_idempotency() -> None:
    call_count = {'plans': 0}

    def counting_override(ctx, instr):
        call_count['plans'] += 1
        return {'action': 'update', 'updates': [{'tool': 'slim_face', 'enabled': True, 'strength': 0.5}], 'question': None}

    def block_high_slim(ctx):
        for u in ctx['plan'].get('updates', []):
            if u.get('tool') == 'slim_face' and u.get('strength', 0) > 0.8:
                return {'block': True, 'reason': '强度超出安全阈值'}
        return None

    def tag_after(ctx, outcome: ToolOutcome) -> ToolOutcome:
        return ToolOutcome(
            output=outcome.output,
            structured_output=outcome.structured_output,
            details={**outcome.details, 'audited_by_hook': True},
            is_error=outcome.is_error,
            terminate=outcome.terminate,
        )

    with tempfile.TemporaryDirectory(prefix='lumina_hooks_') as tmp:
        smile_img, _ = _make_test_images(Path(tmp))
        store = TreeSessionStore(Path(tmp))
        agent = PortraitAgent(
            store=store,
            adapter=RetouchAdapter(mode='mock'),
            planner=MockPlanner(override_fn=counting_override),
            before_tool_hooks=[block_high_slim],
            after_tool_hooks=[tag_after],
        )
        sid = agent.init_session(smile_img)['session_id']

        first = agent.run_turn(sid, '瘦脸', request_id='req-001')
        second = agent.run_turn(sid, '瘦脸', request_id='req-001')
        assert call_count['plans'] == 1
        assert first['replayed'] is False and second['replayed'] is True
        assert first['entry']['ui_details']['audited_by_hook'] is True

        agent.planner = MockPlanner(
            override_fn=lambda ctx, instr: {
                'action': 'update',
                'updates': [{'tool': 'slim_face', 'enabled': True, 'strength': 0.95}],
                'question': None,
            }
        )
        blocked = agent.run_turn(sid, '极端瘦脸')
        assert blocked['entry']['status'] == 'rejected' and blocked['entry']['is_error'] is True
        assert blocked['projection']['state']['slim_face'] == {'enabled': True, 'strength': 0.5}


def test_repeated_clarify_loop_folding() -> None:
    with tempfile.TemporaryDirectory(prefix='lumina_fold_') as tmp:
        smile_img, _ = _make_test_images(Path(tmp))
        store = TreeSessionStore(Path(tmp))
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
        sid = agent.init_session(smile_img)['session_id']
        agent.run_turn(sid, '帮我大眼')
        agent.run_turn(sid, '眼睛小一点')
        assert store.convert_to_llm(sid)['clarify_loop_folded'] is False

        agent.run_turn(sid, '调整大眼，眼睛小一点')
        ctx2 = store.convert_to_llm(sid)
        assert ctx2['clarify_loop_folded'] is True
        assert ctx2['open_signal'] is None
        assert len(ctx2['dialogue']) == 2
        assert len(store.load_entries(sid)) == 3


def test_native_engine_core_operators() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(80, 210, size=(128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[24:104, 24:104] = 1.0

    healed = heal_blemishes(img, np.array([[64.0, 64.0, 4.0]], dtype=np.float32), face_width=80.0)
    assert healed.shape == img.shape and healed.dtype == np.uint8

    smoothed = smooth_skin(to_float(healed), mask, face_width=80.0, strength=0.6)
    assert smoothed.shape == (128, 128, 3) and 0.0 <= float(smoothed.min()) <= float(smoothed.max()) <= 1.0

    bokeh = blur_background(smoothed, mask, strength=0.5)
    mouth = np.zeros((128, 128), dtype=np.float32)
    mouth[70:85, 50:78] = 1.0
    whitened = whiten_teeth(bokeh, mouth, strength=0.5)
    assert whitened.shape == bokeh.shape

    field = new_displacement_field((128, 128))
    field[40:80, 40:80, 0] = 3.5
    warped, scale = apply_displacement(to_uint8(whitened), field)
    assert warped.shape == img.shape and 0.0 < scale <= 1.0


def main() -> None:
    test_tree_branching_and_non_destructive_undo()
    test_teeth_gate_and_context_projection()
    test_hooks_barrier_and_idempotency()
    test_repeated_clarify_loop_folding()
    test_native_engine_core_operators()
    print(json.dumps({
        'unit_tests': 'passed',
        'count': 5,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
