"""Pi-inspired minimal PortraitAgent harness with event streaming, hooks, and Commit-Before-Show."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Callable, Iterator
import uuid

from src.session.tree_store import TreeSessionStore, file_sha256
from src.tools.protocol import ToolOutcome, merge_state, validate_plan_structure
from src.tools.retouch_adapter import RetouchAdapter
from .planners import BasePlanner


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentEvent:
    type: str
    session_id: str
    payload: dict = field(default_factory=dict)
    turn_id: str | None = None
    timestamp: str = field(default_factory=_utc_now)


BeforeToolHook = Callable[[dict], dict | None]
AfterToolHook = Callable[[dict, ToolOutcome], ToolOutcome | None]


class PortraitAgent:
    """Unified single-portrait Agent engine shared by Web Demo, CLI, and batch evaluation."""

    def __init__(
        self,
        store: TreeSessionStore,
        adapter: RetouchAdapter,
        planner: BasePlanner,
        before_tool_hooks: list[BeforeToolHook] | None = None,
        after_tool_hooks: list[AfterToolHook] | None = None,
    ):
        self.store = store
        self.adapter = adapter
        self.planner = planner
        self.before_tool_hooks: list[BeforeToolHook] = list(before_tool_hooks or [])
        self.after_tool_hooks: list[AfterToolHook] = list(after_tool_hooks or [])
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.RLock())

    def init_session(
        self,
        source_image: Path,
        session_id: str | None = None,
        meta: dict | None = None,
    ) -> dict:
        sid = session_id or ('web_' + uuid.uuid4().hex[:16])
        t0 = time.monotonic()
        with self._lock_for(sid):
            tmp_overlay = self.store.sessions_dir / f'.{sid}_analysis.png'
            analysis, analysis_cmd = self.adapter.analyze(Path(source_image), tmp_overlay)
            full_meta = {
                **(meta or {}),
                'planner_mode': self.planner.name,
                'adapter_mode': self.adapter.mode,
                'init_seconds': round(time.monotonic() - t0, 3),
                'analysis_command': analysis_cmd,
            }
            self.store.create_session(sid, source_image, analysis, meta=full_meta)
            if tmp_overlay.is_file():
                tmp_overlay.replace(self.store.session_dir(sid) / 'analysis.png')
            return self.store.project_state(sid)

    def switch_branch(self, session_id: str, leaf_id: str | None) -> dict:
        with self._lock_for(session_id):
            self.store.set_leaf_id(session_id, leaf_id)
            return self.store.project_state(session_id)

    def inspect_session(self, session_id: str, leaf_id: str | None = ...) -> dict:
        with self._lock_for(session_id):
            return self.store.project_state(session_id, leaf_id=leaf_id)

    def run_turn_events(
        self,
        session_id: str,
        instruction: str,
        request_id: str | None = None,
        parent_id: str | None = ...,
    ) -> Iterator[AgentEvent]:
        """Execute one turn, yielding Pi-style lifecycle events with Commit-Before-Show."""
        text = instruction.strip() if isinstance(instruction, str) else ''
        if not 0 < len(text) <= 1000:
            raise ValueError('instruction must contain 1-1000 non-whitespace characters')

        with self._lock_for(session_id):
            # 1. Idempotency check (pi-durable requestId semantics)
            if request_id:
                existing = self.store.find_request(session_id, request_id)
                if existing is not None:
                    proj = self.store.project_state(session_id)
                    yield AgentEvent('turn_end', session_id, {'entry': existing, 'replayed': True, 'projection': proj}, existing['id'])
                    yield AgentEvent('agent_end', session_id, {'replayed': True, 'leaf_id': proj['leaf_id']}, existing['id'])
                    return

            # 2. Resolve branch base (supports branching directly from any historical parent_id!)
            base_leaf = self.store.get_leaf_id(session_id) if parent_id is ... else parent_id
            proj = self.store.project_state(session_id, leaf_id=base_leaf)
            header = proj['header']
            original = Path(header['original'])
            if file_sha256(original) != header['original_sha256']:
                raise RuntimeError('original image checksum changed')

            turn_id = uuid.uuid4().hex[:8]
            started = time.monotonic()
            yield AgentEvent('agent_start', session_id, {'parent_id': base_leaf}, turn_id)
            yield AgentEvent('turn_start', session_id, {'instruction': text, 'parent_id': base_leaf}, turn_id)

            # 3. Project clean LLM context (AgentMessage -> LLM Message boundary)
            llm_ctx = self.store.convert_to_llm(session_id, leaf_id=base_leaf)
            plan_res = self.planner.plan(llm_ctx, text)
            yield AgentEvent(
                'plan_ready',
                session_id,
                {
                    'raw_text': plan_res.raw_text,
                    'strict_json': plan_res.strict_json,
                    'parsed_plan': plan_res.parsed_plan,
                    'plan_seconds': plan_res.plan_seconds,
                    'model': plan_res.model_name,
                },
                turn_id,
            )

            # 4. Validate structure & execute hooks + tool adapter
            state_before = copy.deepcopy(proj['state'])
            open_signal_before = proj['open_signal']
            effective_plan = copy.deepcopy(plan_res.parsed_plan)
            signal_record: dict | None = None

            try:
                if plan_res.parsed_plan is None:
                    raise ValueError('model did not output valid JSON plan')
                validate_plan_structure(plan_res.parsed_plan)

                # Run before_tool_call hook barrier (can block execution cleanly)
                hook_ctx = {
                    'session_id': session_id,
                    'turn_id': turn_id,
                    'instruction': text,
                    'plan': effective_plan,
                    'state_before': state_before,
                    'open_signal': open_signal_before,
                }
                blocked_reason: str | None = None
                for hook in self.before_tool_hooks:
                    decision = hook(hook_ctx)
                    if isinstance(decision, dict) and decision.get('block'):
                        blocked_reason = str(decision.get('reason') or 'blocked by before_tool_call hook')
                        break
                if blocked_reason is not None:
                    raise ValueError(blocked_reason)

                yield AgentEvent('tool_execution_start', session_id, {'action': effective_plan['action']}, turn_id)

                # Teeth visibility gate
                effective_plan, pre_skipped = self.adapter.check_teeth_gate(header['analysis'], effective_plan)

                if effective_plan['action'] == 'clarify':
                    question = effective_plan['question']
                    sig_id = 'sig_' + uuid.uuid4().hex[:8]
                    signal_record = {
                        'signal_id': sig_id,
                        'kind': 'clarify',
                        'status': 'open',
                        'question': question,
                    }
                    outcome = ToolOutcome(
                        output=question,
                        structured_output={'status': 'clarify', 'state': state_before, 'question': question},
                        details={
                            'question': question,
                            'skipped': pre_skipped,
                            'applied': [],
                            'warnings': [],
                            'render_seconds': 0.0,
                        },
                        is_error=False,
                        terminate=True,
                    )
                else:
                    target_state = merge_state(state_before, proj['history_stack'], effective_plan)
                    outcome = self.adapter.execute_render(
                        original=original,
                        target_state=target_state,
                        output_dir=self.store.session_dir(session_id),
                        output_stem=f'node_{turn_id}',
                        action=effective_plan['action'],
                        pre_skipped=pre_skipped,
                    )
                    # Any successful edit/undo/reset closes an earlier open signal cleanly
                    if open_signal_before is not None:
                        signal_record = {
                            **open_signal_before,
                            'status': 'consumed',
                        }
            except Exception as exc:
                outcome = ToolOutcome(
                    output=f'本轮未执行：{exc}',
                    structured_output={'status': 'rejected', 'state': state_before, 'reason': str(exc)},
                    details={
                        'reason': str(exc),
                        'applied': [],
                        'skipped': [],
                        'warnings': [],
                        'render_seconds': 0.0,
                    },
                    is_error=True,
                    terminate=True,
                )

            # Run after_tool_call hooks
            after_ctx = {
                'session_id': session_id,
                'turn_id': turn_id,
                'instruction': text,
                'plan': effective_plan,
                'state_before': state_before,
            }
            for hook in self.after_tool_hooks:
                override = hook(after_ctx, outcome)
                if isinstance(override, ToolOutcome):
                    outcome = override

            yield AgentEvent(
                'tool_execution_end',
                session_id,
                {
                    'status': outcome.structured_output.get('status'),
                    'is_error': outcome.is_error,
                    'terminate': outcome.terminate,
                    'applied': outcome.details.get('applied', []),
                    'skipped': outcome.details.get('skipped', []),
                    'render_seconds': outcome.details.get('render_seconds', 0.0),
                },
                turn_id,
            )

            # 5. Commit-Before-Show: persist immutable tree node before returning to UI
            total_seconds = round(time.monotonic() - started, 3)
            entry_payload = {
                'id': turn_id,
                'parent_id': base_leaf,
                'request_id': request_id,
                'instruction': text,
                'planner': plan_res.model_name,
                'raw_text': plan_res.raw_text,
                'strict_json': plan_res.strict_json,
                'raw_plan': plan_res.parsed_plan,
                'effective_plan': effective_plan,
                'status': outcome.structured_output.get('status', 'rejected'),
                'is_error': outcome.is_error,
                'state_before': state_before,
                'state_after': outcome.structured_output.get('state', state_before),
                'model_output': outcome.output,
                'ui_details': {
                    **outcome.details,
                    'plan_seconds': plan_res.plan_seconds,
                    'total_seconds': total_seconds,
                },
                'signal': signal_record,
            }
            committed = self.store.append_turn(session_id, entry_payload)
            next_proj = self.store.project_state(session_id)

            yield AgentEvent(
                'turn_end',
                session_id,
                {'entry': committed, 'replayed': False, 'projection': next_proj},
                turn_id,
            )
            yield AgentEvent('agent_end', session_id, {'leaf_id': next_proj['leaf_id']}, turn_id)

    def run_turn(
        self,
        session_id: str,
        instruction: str,
        request_id: str | None = None,
        parent_id: str | None = ...,
    ) -> dict:
        final_payload: dict | None = None
        events_log: list[str] = []
        for ev in self.run_turn_events(session_id, instruction, request_id=request_id, parent_id=parent_id):
            events_log.append(ev.type)
            if ev.type == 'turn_end':
                final_payload = ev.payload
        assert final_payload is not None
        return {**final_payload, 'events': events_log}
