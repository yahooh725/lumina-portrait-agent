#!/usr/bin/env python3
"""Lumina Agent local visual Web Demo powered by the unified Pi-inspired `src/` harness.

Zero external web-framework dependencies (Python stdlib HTTP server + static HTML/JS).
Supports three planner backends:
  - six1800 : Frozen Qwen3-VL-2B-Instruct + 1800-sample six-tool LoRA (GPU, default)
  - rules   : Built-in conservative rule planner + real pixel retouching (CPU)
  - mock    : Fast CPU smoke-test mode (no GPU, no retouch subprocess)
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback
import uuid

from PIL import Image

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent
FINAL = BASE / 'final_acceptance_v1'
STATIC_DIR = HERE / 'static'
CANDIDATES = FINAL / 'visual_candidates/images'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.agent import MockPlanner, PortraitAgent, RulePlanner, SixToolLoraPlanner
from src.agent.planners import resolve_default_adapter, resolve_default_base_model, resolve_default_prompt
from src.session import TreeSessionStore
from src.tools import RetouchAdapter, TOOL_META

MAX_UPLOAD = 20 * 1024 * 1024
MAX_PIXELS = 30_000_000
MIME = {'JPEG': ('image/jpeg', '.jpg'), 'PNG': ('image/png', '.png')}

PRESETS = [
    {
        'id': 'teeth_smile',
        'label': '露齿微笑（推荐：多轮牙齿美白/撤销）',
        'tag': '露齿 · CelebAMask-HQ',
        'path': CANDIDATES / 'CelebAMask-HQ/307.jpg',
        'suggested_prompts': [
            '帮我美白牙齿',
            '更明显一点',
            '瘦脸强度调到 0.4',
            '关掉牙齿美白',
            '撤销上一轮',
        ],
    },
    {
        'id': 'closed_mouth_bg',
        'label': '闭嘴人像（推荐：背景虚化 + 闭嘴跳过美白）',
        'tag': '闭嘴 · FFHQ',
        'path': CANDIDATES / 'FFHQ/00244.png',
        'suggested_prompts': [
            '帮我虚化背景',
            '背景虚化减轻一点',
            '帮我美白牙齿',
            '把背景虚化关掉',
            '回到最初',
        ],
    },
    {
        'id': 'slim_portrait',
        'label': '人像瘦脸与大眼',
        'tag': '五官 · CelebAMask-HQ',
        'path': CANDIDATES / 'CelebAMask-HQ/231.jpg',
        'suggested_prompts': [
            '脸小点',
            '大眼调到 0.4',
            '瘦脸再轻一点',
            '只保留瘦脸，大眼关掉',
            '恢复原图',
        ],
    },
    {
        'id': 'skin_acne',
        'label': '祛痘与磨皮组合',
        'tag': '皮肤 · FFHQ',
        'path': CANDIDATES / 'FFHQ/00213.png',
        'suggested_prompts': [
            '帮我祛痘',
            '磨皮强度设为 0.4',
            '祛痘敏感度调到 0.3',
            '撤销上一步',
        ],
    },
    {
        'id': 'smooth_portrait',
        'label': '轻度磨皮人像',
        'tag': '磨皮 · CelebAMask-HQ',
        'path': CANDIDATES / 'CelebAMask-HQ/245.jpg',
        'suggested_prompts': [
            '帮我磨皮',
            '磨皮轻一点',
            '同时把背景虚化调到 0.5',
            '恢复原图',
        ],
    },
    {
        'id': 'eyes_portrait',
        'label': '大眼效果演示',
        'tag': '眼部 · CelebAMask-HQ',
        'path': CANDIDATES / 'CelebAMask-HQ/257.jpg',
        'suggested_prompts': [
            '来点大眼',
            '大眼强度设为 0.3',
            '撤销上一步',
        ],
    },
]


class DemoError(ValueError):
    pass


def validate_image_bytes(data: bytes) -> tuple[str, str]:
    if not data or len(data) > MAX_UPLOAD:
        raise DemoError('图片大小需在 1 字节到 20 MiB 之间')
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in MIME or img.width * img.height > MAX_PIXELS:
                raise DemoError('仅支持 3000 万像素以内的 PNG 或 JPEG 图片')
            img.verify()
            return MIME[img.format]
    except (OSError, ValueError) as exc:
        raise DemoError(f'无效图片文件: {exc}') from exc


def image_media_type(path: Path) -> str:
    return 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'


class DemoEngine:
    """Thin HTTP-facing facade over the unified `src.agent.PortraitAgent` harness."""

    def __init__(self, data_dir: Path, planner_mode: str = 'six1800', prompt_path: Path | None = None):
        if planner_mode not in ('six1800', 'rules', 'mock'):
            raise ValueError('planner_mode must be six1800, rules, or mock')
        self.data_dir = Path(data_dir).resolve()
        self.uploads_dir = self.data_dir / 'uploads'
        self.sessions_dir = self.data_dir / 'sessions'
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('PHOTO_RETOUCH_MODEL_DIR', str(ROOT / 'tool_acceptance/models'))

        self.planner_mode = planner_mode
        if planner_mode == 'six1800':
            planner = SixToolLoraPlanner(prompt_path=prompt_path)
            adapter = RetouchAdapter(mode='real')
        elif planner_mode == 'rules':
            planner = RulePlanner()
            adapter = RetouchAdapter(mode='real')
        else:
            planner = MockPlanner()
            adapter = RetouchAdapter(mode='mock')

        self.store = TreeSessionStore(self.sessions_dir)
        self.agent = PortraitAgent(store=self.store, adapter=adapter, planner=planner)
        self._presets_by_id = {
            item['id']: item for item in PRESETS if Path(item['path']).is_file()
        }

    def preload(self) -> dict:
        return self.agent.planner.preload()

    def health(self) -> dict:
        return {
            'status': 'ok',
            'harness': 'src.agent.PortraitAgent (tree-session)',
            **self.agent.planner.metadata(),
            'tools': TOOL_META,
            'presets_count': len(self._presets_by_id),
        }

    def list_presets(self) -> list[dict]:
        return [
            {
                'id': item['id'],
                'label': item['label'],
                'tag': item['tag'],
                'thumb_url': f"/api/presets/{item['id']}/image",
                'suggested_prompts': item['suggested_prompts'],
            }
            for item in PRESETS
            if item['id'] in self._presets_by_id
        ]

    def preset_path(self, preset_id: str) -> Path:
        item = self._presets_by_id.get(preset_id)
        if not item:
            raise DemoError('未找到该示例图片')
        return Path(item['path'])

    def upload_image(self, data: bytes) -> dict:
        mime, suffix = validate_image_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        image_id = f'up_{digest[:24]}'
        path = self.uploads_dir / f'{image_id}{suffix}'
        if not path.exists():
            tmp = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
            tmp.write_bytes(data)
            tmp.replace(path)
        return {'image_id': image_id, 'mime': mime, 'bytes': len(data)}

    def _resolve_upload(self, image_id: str) -> Path:
        if not re.fullmatch(r'up_[0-9a-f]{24}', image_id):
            raise DemoError('无效的上传图片 ID')
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = self.uploads_dir / f'{image_id}{ext}'
            if candidate.is_file():
                return candidate
        raise DemoError('上传的图片不存在')

    def _validate_sid(self, sid: str) -> None:
        if not re.fullmatch(r'web_[0-9a-f]{16}', sid) or not self.store.exists(sid):
            raise DemoError('会话不存在')

    def _format_projection(self, proj: dict) -> dict:
        sid = proj['session_id']
        header = proj['header']
        meta = header.get('meta') or {}
        original_name = Path(header['original']).name
        current_name = Path(proj['current_image']).name

        turns = []
        for idx, node in enumerate(proj['active_branch'], 1):
            ui = node.get('ui_details') or {}
            img_path = ui.get('image_path')
            img_name = Path(img_path).name if img_path else None
            turns.append({
                'id': node['id'],
                'parent_id': node.get('parent_id'),
                'seq': node.get('seq', idx),
                'turn': idx,
                'instruction': node.get('instruction'),
                'status': node.get('status'),
                'raw_text': node.get('raw_text'),
                'strict_json': node.get('strict_json'),
                'plan': node.get('raw_plan'),
                'effective_plan': node.get('effective_plan'),
                'state_before': node.get('state_before'),
                'state_after': node.get('state_after'),
                'applied': ui.get('applied', []),
                'skipped': ui.get('skipped', []),
                'warnings': ui.get('warnings', []),
                'question': ui.get('question'),
                'reason': ui.get('reason'),
                'image_url': f'/api/sessions/{sid}/images/{img_name}' if img_name else None,
                'plan_seconds': ui.get('plan_seconds'),
                'render_seconds': ui.get('render_seconds'),
                'total_seconds': ui.get('total_seconds'),
            })

        teeth_info = (header.get('analysis', {}).get('people') or [{}])[0].get('teeth_target')
        return {
            'session_id': sid,
            'planner_mode': meta.get('planner_mode', self.planner_mode),
            'preset_id': meta.get('preset_id'),
            'init_seconds': meta.get('init_seconds'),
            'leaf_id': proj['leaf_id'],
            'turn': proj['branch_depth'],
            'total_nodes': proj['total_nodes'],
            'branches': proj['branch_leaves'],
            'open_signal': proj['open_signal'],
            'can_undo': proj['can_undo'],
            'state': proj['state'],
            'original_url': f'/api/sessions/{sid}/images/{original_name}',
            'current_url': f'/api/sessions/{sid}/images/{current_name}',
            'teeth_target': teeth_info,
            'turns': turns,
            'tools': TOOL_META,
        }

    def start_session(self, preset_id: str | None = None, image_id: str | None = None) -> dict:
        if bool(preset_id) == bool(image_id):
            raise DemoError('请提供 preset_id 或 image_id 其中之一')
        source = self.preset_path(preset_id) if preset_id else self._resolve_upload(image_id)
        try:
            proj = self.agent.init_session(source, meta={'preset_id': preset_id})
        except ValueError as exc:
            raise DemoError(f'初始化人像分析失败：{exc}') from exc
        return self._format_projection(proj)

    def inspect_session(self, sid: str) -> dict:
        self._validate_sid(sid)
        return self._format_projection(self.agent.inspect_session(sid))

    def switch_branch(self, sid: str, leaf_id: str | None) -> dict:
        self._validate_sid(sid)
        try:
            proj = self.agent.switch_branch(sid, leaf_id)
        except ValueError as exc:
            raise DemoError(str(exc)) from exc
        return self._format_projection(proj)

    def session_image_path(self, sid: str, filename: str) -> Path:
        self._validate_sid(sid)
        folder = self.store.session_dir(sid)
        if not re.fullmatch(r'(?:original|analysis|turn_\d{3}|node_[0-9a-f]{8})\.(?:png|jpg|jpeg)', filename):
            raise DemoError('无效的图片文件名')
        target = (folder / filename).resolve()
        if target.parent != folder.resolve() or not target.is_file():
            raise DemoError('图片不存在')
        return target

    def run_turn(
        self,
        sid: str,
        instruction: str,
        request_id: str | None = None,
        parent_id: str | None = ...,
    ) -> dict:
        self._validate_sid(sid)
        text = instruction.strip() if isinstance(instruction, str) else ''
        if not 0 < len(text) <= 500:
            raise DemoError('请输入 1–500 字的中文修图指令')
        try:
            res = self.agent.run_turn(sid, text, request_id=request_id, parent_id=parent_id)
        except ValueError as exc:
            raise DemoError(str(exc)) from exc
        payload = self._format_projection(res['projection'])
        payload['last_events'] = res['events']
        payload['replayed'] = res['replayed']
        return payload


def make_handler(engine: DemoEngine, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _reply_json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self._send_bytes(status, 'application/json; charset=utf-8', body)

        def _auth_ok(self) -> bool:
            if not token:
                return True
            if self.headers.get('Authorization') == 'Bearer ' + token:
                return True
            self._reply_json(401, {'error': '未授权访问，请提供 Bearer Token'})
            return False

        def _read_body(self, limit: int) -> bytes:
            raw = self.headers.get('Content-Length')
            if raw is None or not raw.isdecimal() or int(raw) > limit:
                raise DemoError(f'请求体大小无效（上限 {limit} 字节）')
            return self.rfile.read(int(raw))

        def _read_json(self) -> dict:
            try:
                val = json.loads(self._read_body(8192).decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DemoError(f'JSON 解析失败: {exc}') from exc
            if not isinstance(val, dict):
                raise DemoError('JSON 请求体必须为对象')
            return val

        def _dispatch(self, method: str) -> None:
            path = self.path.split('?', 1)[0]
            try:
                if method == 'GET' and path in ('/', '/index.html'):
                    return self._send_bytes(200, 'text/html; charset=utf-8', (STATIC_DIR / 'index.html').read_bytes())
                if not self._auth_ok():
                    return
                if method == 'GET' and path == '/api/health':
                    return self._reply_json(200, engine.health())
                if method == 'GET' and path == '/api/presets':
                    return self._reply_json(200, {'presets': engine.list_presets()})
                m_preset = re.fullmatch(r'/api/presets/([a-z0-9_]+)/image', path)
                if method == 'GET' and m_preset:
                    img = engine.preset_path(m_preset.group(1))
                    return self._send_bytes(200, image_media_type(img), img.read_bytes())
                if method == 'POST' and path == '/api/upload':
                    data = self._read_body(MAX_UPLOAD)
                    return self._reply_json(201, engine.upload_image(data))
                if method == 'POST' and path == '/api/sessions':
                    body = self._read_json()
                    return self._reply_json(
                        201, engine.start_session(body.get('preset_id'), body.get('image_id'))
                    )
                m_sess = re.fullmatch(r'/api/sessions/(web_[0-9a-f]{16})', path)
                if method == 'GET' and m_sess:
                    return self._reply_json(200, engine.inspect_session(m_sess.group(1)))
                m_branch = re.fullmatch(r'/api/sessions/(web_[0-9a-f]{16})/branch', path)
                if method == 'POST' and m_branch:
                    body = self._read_json()
                    if 'leaf_id' not in body:
                        raise DemoError('leaf_id field is required')
                    return self._reply_json(200, engine.switch_branch(m_branch.group(1), body.get('leaf_id')))
                m_turn = re.fullmatch(r'/api/sessions/(web_[0-9a-f]{16})/turn', path)
                if method == 'POST' and m_turn:
                    body = self._read_json()
                    parent_kw = {'parent_id': body['parent_id']} if 'parent_id' in body else {}
                    return self._reply_json(
                        200,
                        engine.run_turn(
                            m_turn.group(1),
                            body.get('instruction', ''),
                            request_id=body.get('request_id'),
                            **parent_kw,
                        ),
                    )
                m_img = re.fullmatch(r'/api/sessions/(web_[0-9a-f]{16})/images/([A-Za-z0-9_.-]+)', path)
                if method == 'GET' and m_img:
                    img = engine.session_image_path(m_img.group(1), m_img.group(2))
                    return self._send_bytes(200, image_media_type(img), img.read_bytes())
                return self._reply_json(404, {'error': 'route not found'})
            except DemoError as exc:
                return self._reply_json(400, {'error': str(exc)})
            except Exception as exc:
                traceback.print_exc()
                return self._reply_json(500, {'error': f'{type(exc).__name__}: {exc}'})

        def do_GET(self):
            self._dispatch('GET')

        def do_POST(self):
            self._dispatch('POST')

        def log_message(self, fmt: str, *args) -> None:
            print(f'[web_demo] {self.address_string()} {fmt % args}', flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--planner', choices=['six1800', 'rules', 'mock'], default='six1800')
    parser.add_argument('--data-dir', type=Path, default=HERE / 'runtime_demo')
    parser.add_argument('--preload', action='store_true', help='Load six1800 LoRA on startup')
    parser.add_argument('--preflight', action='store_true', help='Verify assets without starting server or GPU')
    args = parser.parse_args()

    if args.preflight:
        base_model = resolve_default_base_model()
        adapter_dir = resolve_default_adapter()
        prompt_file = resolve_default_prompt()
        checks = {
            'static_index': (STATIC_DIR / 'index.html').is_file(),
            'base_model': base_model,
            'model_dir': Path(base_model).is_dir(),
            'adapter_weights': (adapter_dir / 'adapter_model.safetensors').is_file(),
            'prompt': prompt_file.is_file(),
            'presets_available': sum(1 for item in PRESETS if Path(item['path']).is_file()),
            'planner_mode': args.planner,
            'harness': 'src.agent.PortraitAgent',
            'engine': 'lumina-native-v1',
        }
        ok = checks['static_index'] and checks['prompt']
        if args.planner == 'six1800':
            ok = ok and checks['adapter_weights']
        print(json.dumps({'status': 'ready' if ok else 'failed', **checks}, ensure_ascii=False, indent=2))
        raise SystemExit(0 if ok else 1)

    token = os.environ.get('PORTRAIT_DEMO_TOKEN')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not token:
        raise SystemExit('非本机地址监听请设置环境变量 PORTRAIT_DEMO_TOKEN')

    engine = DemoEngine(args.data_dir, planner_mode=args.planner)
    if args.preload and args.planner == 'six1800':
        info = engine.preload()
        print(json.dumps({'event': 'model_preloaded', **info}, ensure_ascii=False), flush=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, token))
    print(json.dumps({
        'listening': f'http://{args.host}:{args.port}',
        'planner_mode': args.planner,
        'harness': 'src.agent.PortraitAgent',
        'data_dir': str(engine.data_dir),
        'presets': len(engine.list_presets()),
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
