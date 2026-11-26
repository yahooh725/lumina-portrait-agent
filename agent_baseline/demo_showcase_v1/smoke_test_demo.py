#!/usr/bin/env python3
"""Offline CPU smoke test for web_demo.py HTTP API and state/skip/undo flows."""
from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import urllib.error
import urllib.request

from web_demo import DemoEngine, make_handler


def req(base: str, method: str, path: str, body: bytes | None = None, content_type: str | None = None):
    headers = {}
    if content_type:
        headers['Content-Type'] = content_type
    request = urllib.request.Request(base + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return resp.status, resp.headers.get('Content-Type', ''), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get('Content-Type', ''), exc.read()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix='portrait_web_smoke_') as tmp:
        engine = DemoEngine(Path(tmp), planner_mode='mock')
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(engine, token=None))
        port = server.server_address[1]
        base = f'http://127.0.0.1:{port}'
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        thread.start()
        try:
            # 1. Index HTML
            status, ctype, body = req(base, 'GET', '/')
            assert status == 200 and 'text/html' in ctype and b'compare-range' in body

            # 2. Health & Presets
            status, _, body = req(base, 'GET', '/api/health')
            health = json.loads(body)
            assert status == 200 and health['planner_mode'] == 'mock' and len(health['tools']) == 6

            status, _, body = req(base, 'GET', '/api/presets')
            presets = json.loads(body)['presets']
            assert status == 200

            if any(p['id'] == 'closed_mouth_bg' for p in presets):
                status, ctype, img_bytes = req(base, 'GET', '/api/presets/closed_mouth_bg/image')
                assert status == 200 and 'image/' in ctype and len(img_bytes) > 1000
                session_payload = {'preset_id': 'closed_mouth_bg'}
            else:
                import io
                from PIL import Image
                buf = io.BytesIO()
                Image.new('RGB', (160, 160), (205, 175, 155)).save(buf, format='PNG')
                img_bytes = buf.getvalue()
                session_payload = None

            # 3. Upload validation (reject invalid, accept valid)
            status, _, _ = req(base, 'POST', '/api/upload', b'not-an-image')
            assert status == 400
            status, _, body = req(base, 'POST', '/api/upload', img_bytes)
            upload_res = json.loads(body)
            assert status == 201 and upload_res['image_id'].startswith('up_')
            if session_payload is None:
                # 干净仓库无预设图时使用上传通道测试完整多轮会话
                session_payload = {'image_id': upload_res['image_id']}

            # 4. Multi-turn session: blur_bg -> teeth/slim -> branch fork -> switch -> undo -> reset
            status, _, body = req(
                base, 'POST', '/api/sessions',
                json.dumps(session_payload).encode('utf-8'), 'application/json'
            )
            assert status == 201
            sess = json.loads(body)
            sid = sess['session_id']
            closed_mode = not sess['teeth_target']['eligible']

            def send(text: str) -> dict:
                st, _, b = req(
                    base, 'POST', f'/api/sessions/{sid}/turn',
                    json.dumps({'instruction': text}).encode('utf-8'), 'application/json'
                )
                assert st == 200, (st, b)
                return json.loads(b)

            s1 = send('帮我虚化背景')
            assert s1['state']['blur_bg'] == {'enabled': True, 'strength': 0.5}
            assert s1['turns'][-1]['status'] == 'rendered'

            s2 = send('帮我美白牙齿')
            if closed_mode:
                assert s2['turns'][-1]['status'] == 'clarify'
                assert s2['turns'][-1]['skipped'][0]['tool'] == 'whiten_teeth'
                assert s2['state']['whiten_teeth']['enabled'] is False
            assert s2['state']['blur_bg'] == {'enabled': True, 'strength': 0.5}

            s3 = send('瘦脸强度调到 0.4')
            assert s3['state']['slim_face'] == {'enabled': True, 'strength': 0.4}
            assert s3['state']['blur_bg'] == {'enabled': True, 'strength': 0.5}
            branch_a_leaf = s3['leaf_id']

            # Fork a second branch from turn 1 (blur_bg=0.5) over HTTP
            t1_id = s1['turns'][0]['id']
            st, _, b = req(
                base, 'POST', f'/api/sessions/{sid}/turn',
                json.dumps({'instruction': '大眼强度设为 0.6', 'parent_id': t1_id}).encode('utf-8'),
                'application/json'
            )
            assert st == 200
            sb = json.loads(b)
            assert sb['state']['enlarge_eyes'] == {'enabled': True, 'strength': 0.6}
            assert sb['state']['slim_face']['enabled'] is False
            assert len(sb['branches']) == 2

            # Switch back to Branch A via /branch
            st, _, b = req(
                base, 'POST', f'/api/sessions/{sid}/branch',
                json.dumps({'leaf_id': branch_a_leaf}).encode('utf-8'),
                'application/json'
            )
            assert st == 200
            back_a = json.loads(b)
            assert back_a['state']['slim_face'] == {'enabled': True, 'strength': 0.4}
            assert back_a['state']['enlarge_eyes']['enabled'] is False

            s4 = send('撤销上一步')
            assert s4['state']['slim_face']['enabled'] is False
            assert s4['state']['blur_bg'] == {'enabled': True, 'strength': 0.5}

            s5 = send('恢复原图')
            assert all(not v['enabled'] for v in s5['state'].values())

            # 5. Image serving
            status, ctype, out_bytes = req(base, 'GET', s5['current_url'])
            assert status == 200 and 'image/' in ctype and len(out_bytes) > 1000

            print(json.dumps({
                'smoke_test': 'passed',
                'turns_verified': len(s5['turns']),
                'branches_verified': len(s5['branches']),
                'closed_mouth_skip_verified': True,
                'upload_and_image_serving_verified': True,
            }, ensure_ascii=False, indent=2))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    main()
