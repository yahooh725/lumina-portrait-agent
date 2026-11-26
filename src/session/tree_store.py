"""Append-only tree-structured JSONL session store (inspired by Pi's SessionManager).

Design invariants:
- `events.jsonl` is strictly append-only; line 1 is `session_header`, all subsequent lines are immutable `turn` nodes.
- Each turn node has `id` and `parent_id` (None for root-level turns), forming a tree that supports
  in-place branching without erasing earlier edit paths.
- `head.json` atomically stores the active `leaf_id` pointer so branch switching never rewrites `events.jsonl`.
- `convert_to_llm()` projects only clean model-facing history + open signals along the active branch,
  filtering out UI details, timings, and closed/cancelled signals.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = json.loads((ROOT / 'configs/tool_registry.json').read_text(encoding='utf-8'))
INITIAL_STATE: dict[str, dict] = _REGISTRY['initial_state']


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    tmp = path.with_name(f'{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


class TreeSessionStore:
    """Single-source-of-truth append-only tree session persistence."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir).resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', session_id):
            raise ValueError('invalid session_id')
        return self.sessions_dir / session_id

    def exists(self, session_id: str) -> bool:
        try:
            folder = self.session_dir(session_id)
        except ValueError:
            return False
        return (folder / 'events.jsonl').is_file()

    def create_session(
        self,
        session_id: str,
        source_image: Path,
        analysis: dict,
        meta: dict | None = None,
    ) -> dict:
        source = Path(source_image).resolve()
        if not source.is_file() or source.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise ValueError('source image must be an existing PNG or JPEG')
        folder = self.session_dir(session_id)
        if folder.exists():
            raise ValueError(f'session already exists: {session_id}')
        folder.mkdir(parents=True)
        original = folder / ('original' + source.suffix.lower())
        shutil.copy2(source, original)
        header = {
            'type': 'session_header',
            'version': 1,
            'session_id': session_id,
            'created_utc': utc_now(),
            'source': str(source),
            'original': str(original),
            'original_sha256': file_sha256(original),
            'analysis': analysis,
            'meta': meta or {},
        }
        (folder / 'events.jsonl').write_text(
            json.dumps(header, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8',
        )
        _atomic_write_json(folder / 'head.json', {'leaf_id': None, 'updated_utc': utc_now()})
        return header

    def _read_all_lines(self, session_id: str) -> tuple[dict, list[dict]]:
        folder = self.session_dir(session_id)
        events_file = folder / 'events.jsonl'
        if not events_file.is_file():
            raise FileNotFoundError(f'session not found: {session_id}')
        lines = [json.loads(line) for line in events_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not lines or lines[0].get('type') != 'session_header':
            raise ValueError(f'corrupted session header: {session_id}')
        header = lines[0]
        turns = [item for item in lines[1:] if item.get('type') == 'turn']
        return header, turns

    def load_header(self, session_id: str) -> dict:
        header, _ = self._read_all_lines(session_id)
        return header

    def load_entries(self, session_id: str) -> list[dict]:
        _, turns = self._read_all_lines(session_id)
        return turns

    def get_leaf_id(self, session_id: str) -> str | None:
        head_file = self.session_dir(session_id) / 'head.json'
        if not head_file.is_file():
            turns = self.load_entries(session_id)
            return turns[-1]['id'] if turns else None
        return json.loads(head_file.read_text(encoding='utf-8')).get('leaf_id')

    def set_leaf_id(self, session_id: str, leaf_id: str | None) -> str | None:
        _, turns = self._read_all_lines(session_id)
        if leaf_id is not None and not any(item['id'] == leaf_id for item in turns):
            raise ValueError(f'unknown entry id: {leaf_id}')
        _atomic_write_json(
            self.session_dir(session_id) / 'head.json',
            {'leaf_id': leaf_id, 'updated_utc': utc_now()},
        )
        return leaf_id

    def find_request(self, session_id: str, request_id: str) -> dict | None:
        if not request_id:
            return None
        for entry in self.load_entries(session_id):
            if entry.get('request_id') == request_id:
                return entry
        return None

    def append_turn(self, session_id: str, entry: dict) -> dict:
        """Atomically append an immutable turn entry and advance head.json (Commit-Before-Show)."""
        folder = self.session_dir(session_id)
        header, turns = self._read_all_lines(session_id)
        _ = header
        parent_id = entry.get('parent_id')
        if parent_id is not None and not any(item['id'] == parent_id for item in turns):
            raise ValueError(f'parent_id not found in session tree: {parent_id}')
        record = {
            'type': 'turn',
            'seq': len(turns) + 1,
            'timestamp': utc_now(),
            **entry,
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
        with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line)
            stream.flush()
        _atomic_write_json(
            folder / 'head.json',
            {'leaf_id': record['id'], 'updated_utc': record['timestamp']},
        )
        return record

    def active_branch(self, session_id: str, leaf_id: str | None = ...) -> list[dict]:
        """Walk parent_id links from leaf to root, returning root->leaf ordered entries."""
        _, turns = self._read_all_lines(session_id)
        target_leaf = self.get_leaf_id(session_id) if leaf_id is ... else leaf_id
        if target_leaf is None:
            return []
        by_id = {item['id']: item for item in turns}
        path: list[dict] = []
        cursor: str | None = target_leaf
        seen: set[str] = set()
        while cursor is not None:
            if cursor in seen:
                raise ValueError('cycle detected in session tree')
            seen.add(cursor)
            node = by_id.get(cursor)
            if node is None:
                raise ValueError(f'dangling parent_id in session tree: {cursor}')
            path.append(node)
            cursor = node.get('parent_id')
        path.reverse()
        return path

    def project_state(self, session_id: str, leaf_id: str | None = ...) -> dict:
        """Reconstruct canonical 6-tool state, undo stack, active image, open signal, and tree metadata."""
        header, turns = self._read_all_lines(session_id)
        resolved_leaf = self.get_leaf_id(session_id) if leaf_id is ... else leaf_id
        branch = self.active_branch(session_id, leaf_id=resolved_leaf)

        state = copy.deepcopy(INITIAL_STATE)
        history_stack: list[dict] = []
        current_image = header['original']
        open_signal: dict | None = None

        for node in branch:
            status = node.get('status')
            if status == 'rendered':
                action = (node.get('effective_plan') or node.get('raw_plan') or {}).get('action')
                if action == 'undo':
                    if history_stack:
                        history_stack.pop()
                else:
                    history_stack.append(copy.deepcopy(state))
                state = copy.deepcopy(node['state_after'])
                if node.get('ui_details', {}).get('image_path'):
                    current_image = node['ui_details']['image_path']
            sig = node.get('signal')
            if isinstance(sig, dict):
                if sig.get('status') == 'open':
                    open_signal = copy.deepcopy(sig)
                elif sig.get('status') in ('consumed', 'cancelled'):
                    open_signal = None

        # Compute tree leaves (nodes with no children) for branch awareness
        parent_ids = {item.get('parent_id') for item in turns if item.get('parent_id') is not None}
        leaves = [
            {
                'id': item['id'],
                'seq': item['seq'],
                'instruction': item.get('instruction'),
                'status': item.get('status'),
                'is_active': item['id'] == resolved_leaf,
            }
            for item in turns
            if item['id'] not in parent_ids
        ]

        return {
            'session_id': session_id,
            'header': header,
            'leaf_id': resolved_leaf,
            'state': state,
            'history_stack': history_stack,
            'can_undo': len(history_stack) > 0,
            'current_image': current_image,
            'open_signal': open_signal,
            'active_branch': branch,
            'all_nodes': turns,
            'branch_leaves': leaves,
            'branch_depth': len(branch),
            'total_nodes': len(turns),
        }

    def convert_to_llm(
        self,
        session_id: str,
        leaf_id: str | None = ...,
        max_dialogue_messages: int = 8,
    ) -> dict:
        """Project active branch into clean model-facing context (AgentMessage -> LLM Message boundary).

        Pi-inspired projection rules (purely structural, zero Chinese keyword rules):
        1. Never leaks UI details, timing telemetry, or closed/cancelled signals into model context.
        2. Omits stale clarify/rejected turns that were already superseded by a later rendered edit.
        3. Breaks 2B greedy-decoding repetition attractors: if the same clarify question repeated
           consecutively (>= 2 times) at the tail of the branch, omits the stuck clarify echo from
           the model-facing dialogue while preserving all raw turns intact in `events.jsonl` and UI.
        """
        proj = self.project_state(session_id, leaf_id=leaf_id)
        branch = proj['active_branch']
        dialogue: list[dict[str, str]] = []
        recent_edit: dict | None = None

        # Find index of the last rendered turn so resolved historical clarifies don't pollute context
        last_rendered_idx = -1
        for idx, node in enumerate(branch):
            if node.get('status') == 'rendered':
                last_rendered_idx = idx

        # Check trailing clarify streak after last_rendered_idx
        trailing = branch[last_rendered_idx + 1:]
        trailing_clarifies = [n for n in trailing if n.get('status') == 'clarify']
        repeated_clarify_loop = (
            len(trailing_clarifies) >= 2
            and trailing_clarifies[-1].get('model_output') == trailing_clarifies[-2].get('model_output')
        )

        for idx, node in enumerate(branch):
            status = node.get('status')
            if status == 'rendered':
                dialogue.append({'role': 'user', 'content': node.get('instruction', '')})
                dialogue.append({'role': 'assistant', 'content': node.get('model_output', '')})
            elif idx > last_rendered_idx and status == 'clarify':
                if repeated_clarify_loop:
                    # Drop stuck repeated clarify echo from LLM projection to break repetition loop
                    continue
                if node is trailing_clarifies[-1]:
                    dialogue.append({'role': 'user', 'content': node.get('instruction', '')})
                    dialogue.append({'role': 'assistant', 'content': node.get('model_output', '')})

        for node in reversed(branch):
            if node.get('status') != 'rendered':
                continue
            plan = node.get('effective_plan') or node.get('raw_plan') or {}
            if plan.get('action') in ('undo', 'reset'):
                break
            if plan.get('action') != 'update':
                continue
            before = node.get('state_before') or {}
            after = node.get('state_after') or {}
            changed = [
                {'tool': tool, 'before': before.get(tool), 'after': value}
                for tool, value in after.items()
                if before.get(tool) != value
            ]
            if changed:
                recent_edit = {
                    'entry_id': node['id'],
                    'instruction': node.get('instruction'),
                    'changed': changed,
                }
                break

        return {
            'original': proj['header']['original'],
            'state': proj['state'],
            'dialogue': dialogue[-max_dialogue_messages:],
            'recent_edit': recent_edit,
            'open_signal': None if repeated_clarify_loop else proj['open_signal'],
            'clarify_loop_folded': repeated_clarify_loop,
        }
