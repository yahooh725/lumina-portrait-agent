"""六工具原生像素精修适配层：接入 Lumina 原生进程内引擎、露齿门控与三通道返回协议。"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from src.engine import LuminaRetouchEngine, get_default_engine
from src.session.tree_store import file_sha256
from .protocol import INITIAL_STATE, TOOL_PARAMS, ToolOutcome


class RetouchAdapter:
    """将结构化六工具参数安全映射至 Lumina 原生人像精修引擎（进程内执行，无子进程冷启动开销）。"""

    def __init__(self, mode: str = 'real', engine: LuminaRetouchEngine | None = None):
        if mode not in ('real', 'mock'):
            raise ValueError('RetouchAdapter mode 必须为 real 或 mock')
        self.mode = mode
        self._engine = engine

    def _get_engine(self) -> LuminaRetouchEngine:
        return self._engine or get_default_engine()

    def analyze(self, image_path: Path, overlay_path: Path) -> tuple[dict, dict]:
        if self.mode == 'mock':
            closed = '00244' in image_path.name or 'closed' in image_path.name.lower()
            analysis = {
                'engine': 'lumina-native-mock',
                'people': [{
                    'id': 0,
                    'person': 0,
                    'has_face': True,
                    'teeth_target': {
                        'eligible': not closed,
                        'opening_ratio': 0.002 if closed else 0.128,
                        'candidate_fraction': 0.069 if closed else 0.42,
                        'suspected_occlusion': False,
                    },
                }],
                'mock': True,
            }
            return analysis, {'engine': 'lumina-native-mock', 'mock': True, 'seconds': 0.01}

        analysis, meta = self._get_engine().analyze(image_path, overlay_path=overlay_path)
        people = analysis.get('people', [])
        if len(people) != 1 or not people[0].get('has_face'):
            raise ValueError('当前仅支持包含清晰单人人脸的照片，请换一张单人人像重试。')
        return analysis, meta

    def check_teeth_gate(self, analysis: dict, plan: dict) -> tuple[dict, list[dict]]:
        """基于实测露齿资格门控牙齿美白，闭嘴或遮挡时严格跳过、绝不伪造成功。"""
        if plan.get('action') != 'update':
            return plan, []
        wants_teeth = any(
            u['tool'] == 'whiten_teeth' and u['enabled']
            for u in plan.get('updates', [])
        )
        if not wants_teeth:
            return plan, []
        teeth = (analysis.get('people') or [{}])[0].get('teeth_target') or {}
        if teeth.get('eligible', False):
            return plan, []
        skipped = [{
            'tool': 'whiten_teeth',
            'reason': '未确认到可处理的露齿区域',
            'check': teeth,
        }]
        filtered = copy.deepcopy(plan)
        filtered['updates'] = [u for u in filtered['updates'] if u['tool'] != 'whiten_teeth']
        if not filtered['updates']:
            return {
                'action': 'clarify',
                'updates': [],
                'question': '未确认到可处理的露齿区域，牙齿美白未执行。请换一张露齿照片，或继续调整其他项目。',
            }, skipped
        return filtered, skipped

    def execute_render(
        self,
        original: Path,
        target_state: dict,
        output_dir: Path,
        output_stem: str,
        action: str,
        pre_skipped: list[dict] | None = None,
    ) -> ToolOutcome:
        """从原图 + 完整当前参数重新渲染，并返回三通道 ToolOutcome。"""
        state = copy.deepcopy(target_state)
        skipped = list(pre_skipped or [])
        active = [(tool, item) for tool, item in state.items() if item.get('enabled')]
        suffix = '.png' if active else original.suffix.lower()
        target_path = output_dir / f'{output_stem}{suffix}'

        if not active:
            shutil.copyfile(original, target_path)
            applied: list[str] = []
            warnings: list[str] = []
            command_meta: dict | None = {'engine': 'lumina-native-v1', 'in_process': True, 'seconds': 0.0}
            render_seconds = 0.0
        elif self.mode == 'mock':
            shutil.copyfile(original, target_path)
            applied = [
                f"{tool.replace('_', '-')} {item[TOOL_PARAMS[tool]]} (mock)"
                for tool, item in active
            ]
            warnings = []
            command_meta = {'engine': 'lumina-native-mock', 'mock': True, 'exit_code': 0, 'seconds': 0.02}
            render_seconds = 0.02
        else:
            tool_res, command_meta = self._get_engine().edit(
                original,
                target_path,
                slim_face=float(state['slim_face']['strength']) if state['slim_face']['enabled'] else 0.0,
                enlarge_eyes=float(state['enlarge_eyes']['strength']) if state['enlarge_eyes']['enabled'] else 0.0,
                remove_acne=float(state['remove_acne']['sensitivity']) if state['remove_acne']['enabled'] else None,
                smooth_skin_strength=float(state['smooth_skin']['strength']) if state['smooth_skin']['enabled'] else 0.0,
                blur_bg=float(state['blur_bg']['strength']) if state['blur_bg']['enabled'] else 0.0,
                whiten_teeth_strength=float(state['whiten_teeth']['strength']) if state['whiten_teeth']['enabled'] else 0.0,
                person_index=0,
            )
            if tool_res.get('people_detected') != 1 or tool_res.get('targets') != [0]:
                raise RuntimeError('单人目标校验失败，渲染已中止')
            applied = tool_res.get('applied', [])
            warnings = tool_res.get('warnings', [])
            render_seconds = command_meta.get('seconds', 0.0)
            if tool_res.get('skipped'):
                skipped.extend(tool_res['skipped'])
                for sk in tool_res['skipped']:
                    if sk.get('tool') == 'whiten_teeth':
                        state['whiten_teeth'] = copy.deepcopy(INITIAL_STATE['whiten_teeth'])

        out_sha = file_sha256(target_path)
        if action == 'undo':
            model_msg = '已撤销上一步。'
        elif action == 'reset':
            model_msg = '已回到最初设置。'
        else:
            model_msg = '已完成本轮调整。' + (
                '牙齿美白未执行：未确认到露齿区域。' if skipped else ''
            )

        return ToolOutcome(
            output=model_msg,
            structured_output={
                'status': 'rendered',
                'state': state,
                'image_path': str(target_path),
                'output_sha256': out_sha,
            },
            details={
                'applied': applied,
                'skipped': skipped,
                'warnings': warnings,
                'image_path': str(target_path),
                'output_sha256': out_sha,
                'command': command_meta,
                'render_seconds': render_seconds,
            },
            is_error=False,
            terminate=True,
        )
