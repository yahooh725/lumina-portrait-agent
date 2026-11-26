#!/usr/bin/env python3
"""Lumina Web 服务 HTTP API 端到端冒烟测试（纯 CPU，零外部数据依赖）。"""
from __future__ import annotations

from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import urllib.error
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web.server import LuminaWebApp, make_handler


def req(base: str, method: str, path: str, body: bytes | None = None, content_type: str | None = None):
    headers = {'Content-Type': content_type} if content_type else {}
    request = urllib.request.Request(base + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return resp.status, resp.headers.get('Content-Type', ''), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get('Content-Type', ''), exc.read()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='lumina_web_test_') as tmp:
        app_engine = LuminaWebApp(Path(tmp), planner_mode='mock')
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app_engine, token=None))
        base = f'http://127.0.0.1:{server.server_address[1]}'
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        thread.start()
        try:
            # 1. 首页与健康检查
            status, ctype, body = req(base, 'GET', '/')
            assert status == 200 and 'text/html' in ctype and b'compare-range' in body

            status, _, body = req(base, 'GET', '/api/health')
            health = json.loads(body)
            assert status == 200 and health['status'] == 'ok' and len(health['tools']) == 6

            # 2. 内置示例图与上传校验
            status, _, body = req(base, 'GET', '/api/presets')
            presets = json.loads(body)['presets']
            assert status == 200 and len(presets) >= 1

            status, ctype, img_bytes = req(base, 'GET', presets[0]['thumb_url'])
            assert status == 200 and 'image/' in ctype and len(img_bytes) > 1000

            status, _, _ = req(base, 'POST', '/api/upload', b'not-an-image')
            assert status == 400

            buf = io.BytesIO()
            Image.new('RGB', (160, 160), (200, 175, 160)).save(buf, format='PNG')
            status, _, body = req(base, 'POST', '/api/upload', buf.getvalue())
            assert status == 201 and json.loads(body)['image_id'].startswith('up_')

            # 3. 多轮对话、原地分叉切换、撤销与重置
            status, _, body = req(
                base, 'POST', '/api/sessions',
                json.dumps({'preset_id': presets[0]['id']}).encode('utf-8'), 'application/json',
            )
            assert status == 201
            sid = json.loads(body)['session_id']

            def send(text: str, parent_id: str | None = None) -> dict:
                payload = {'instruction': text}
                if parent_id is not None:
                    payload['parent_id'] = parent_id
                st, _, b = req(
                    base, 'POST', f'/api/sessions/{sid}/turn',
                    json.dumps(payload).encode('utf-8'), 'application/json',
                )
                assert st == 200, (st, b)
                return json.loads(b)

            s1 = send('帮我虚化背景')
            assert s1['state']['blur_bg'] == {'enabled': True, 'strength': 0.5}
            t1_id = s1['turns'][0]['id']

            s2a = send('瘦脸强度调到 0.4')
            assert s2a['state']['slim_face'] == {'enabled': True, 'strength': 0.4}
            branch_a = s2a['leaf_id']

            # 从第 1 轮原地派生方案 B
            s2b = send('大眼强度设为 0.6', parent_id=t1_id)
            assert s2b['state']['enlarge_eyes'] == {'enabled': True, 'strength': 0.6}
            assert s2b['state']['slim_face']['enabled'] is False
            assert len(s2b['branches']) == 2

            # 切回方案 A 并验证撤销与恢复原图
            st, _, b = req(
                base, 'POST', f'/api/sessions/{sid}/branch',
                json.dumps({'leaf_id': branch_a}).encode('utf-8'), 'application/json',
            )
            assert st == 200 and json.loads(b)['state']['slim_face']['strength'] == 0.4

            s_undo = send('撤销上一步')
            assert s_undo['state']['slim_face']['enabled'] is False
            assert s_undo['state']['blur_bg']['enabled'] is True

            s_reset = send('恢复原图')
            assert all(not v['enabled'] for v in s_reset['state'].values())

            status, ctype, out_bytes = req(base, 'GET', s_reset['current_url'])
            assert status == 200 and 'image/' in ctype and len(out_bytes) > 1000

            print(json.dumps({
                'web_api_test': 'passed',
                'turns_verified': len(s_reset['turns']),
                'branches_verified': len(s_reset['branches']),
            }, ensure_ascii=False, indent=2))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    main()
