"""Lumina 原生六工具渲染管线（进程内持久化引擎 + 独立 CLI 入口）。

固定伪影最小化执行顺序：
1. 祛痘（先去瑕疵，避免磨皮或几何变形扩散污点）
2. 磨皮（Lab 高低频肤质平滑）
3. 牙齿美白（含露齿可见性门控，闭嘴自动跳过）
4. 背景虚化（引导细化人像抠图 + 归一化散景）
5. 几何重塑（瘦脸 + 旋转椭圆大眼合并到同一位移场，单次三次样条重采样）
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .analyzer import (
    LIPS_INNER,
    PortraitAnalyzer,
    PortraitScene,
    polygon_mask,
    read_image_bgr,
    write_image_bgr,
)
from .geometry import add_enlarge_eyes, add_slim_face, apply_displacement, new_displacement_field
from .optics import blur_background, check_teeth_eligibility, whiten_teeth
from .skin import find_blemishes, heal_blemishes, smooth_skin, to_float, to_uint8


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _draw_analysis_overlay(img_bgr: np.ndarray, scene: PortraitScene) -> np.ndarray:
    out = img_bgr.copy()
    thick = max(2, int(np.hypot(*img_bgr.shape[:2]) / 600))
    for person in scene.people:
        contours, _ = cv2.findContours(
            (person.mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (220, 145, 70), thick)
        if person.face is not None:
            hull = cv2.convexHull(np.round(person.face).astype(np.int32))
            cv2.polylines(out, [hull], True, (90, 200, 120), max(1, thick // 2))
    return out


class LuminaRetouchEngine:
    """进程内统一原生修图引擎（线程安全、带按图分析缓存）。"""

    def __init__(self, cache_size: int = 16):
        self.analyzer = PortraitAnalyzer(cache_size=cache_size)

    def analyze(self, image_path: str | Path, overlay_path: str | Path | None = None) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        p = Path(image_path)
        sha = _file_sha256(p)
        img = read_image_bgr(p)
        scene, cache_hit = self.analyzer.analyze_bgr(img, image_sha256=sha)
        img_f = to_float(img)
        h, w = img.shape[:2]

        people_info = []
        for person in scene.people:
            entry: dict = {
                'id': person.id,
                'person': person.id,
                'bbox': list(person.bbox),
                'has_face': person.face is not None,
            }
            if person.face is not None:
                fw = person.face_width()
                mouth = polygon_mask(person.face[LIPS_INNER], (h, w))
                entry['face_width_px'] = round(fw)
                entry['teeth_target'] = check_teeth_eligibility(img_f, person.face, mouth, fw)
                spots = find_blemishes(img, scene.face_skin_for_blemish(person), fw, sensitivity=0.5)
                entry['blemishes_detected'] = int(len(spots))
            people_info.append(entry)

        if overlay_path is not None:
            write_image_bgr(overlay_path, _draw_analysis_overlay(img, scene), quality=90)

        elapsed = round(time.perf_counter() - t0, 3)
        analysis = {
            'engine': 'lumina-native-v1',
            'image': str(p),
            'size': [w, h],
            'people': people_info,
            'overlay': str(overlay_path) if overlay_path else None,
        }
        meta = {
            'engine': 'lumina-native-v1',
            'in_process': True,
            'cache_hit': cache_hit,
            'seconds': elapsed,
        }
        return analysis, meta

    def edit(
        self,
        image_path: str | Path,
        output_path: str | Path,
        *,
        slim_face: float = 0.0,
        enlarge_eyes: float = 0.0,
        remove_acne: float | None = None,
        smooth_skin_strength: float = 0.0,
        blur_bg: float = 0.0,
        whiten_teeth_strength: float = 0.0,
        person_index: int = 0,
        quality: int = 95,
    ) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        p = Path(image_path)
        sha = _file_sha256(p)
        original = read_image_bgr(p)
        img = original.copy()
        h, w = img.shape[:2]

        t_det = time.perf_counter()
        scene, cache_hit = self.analyzer.analyze_bgr(img, image_sha256=sha)
        detect_ms = round((time.perf_counter() - t_det) * 1000, 1)

        applied: list[str] = []
        warnings: list[str] = []
        skipped: list[dict] = []

        if not scene.people:
            raise ValueError('未检测到有效单人人像')
        if person_index < 0 or person_index >= len(scene.people):
            raise ValueError(f'人物索引越界：{person_index}（检测到 {len(scene.people)} 人）')
        target = scene.people[person_index]

        # 1. 祛痘（sensitivity=0.0 仍视为开启，由外层 enabled 控制）
        if remove_acne is not None and target.face is not None:
            fw = target.face_width()
            skin_for_acne = scene.face_skin_for_blemish(target)
            spots = find_blemishes(img, skin_for_acne, fw, sensitivity=float(remove_acne))
            img = heal_blemishes(img, spots, fw)
            applied.append(f'remove-acne sensitivity={remove_acne} healed={len(spots)}')

        img_f = to_float(img)
        original_f = to_float(original)

        # 2. Lab 高低频磨皮
        if smooth_skin_strength > 0.0:
            fw = target.face_width() or max(20.0, 0.4 * (target.bbox[2] - target.bbox[0]))
            skin_mask = scene.retouch_skin_mask(target)
            img_f = smooth_skin(img_f, skin_mask, fw, float(smooth_skin_strength))
            applied.append(f'smooth-skin {smooth_skin_strength}')

        # 3. 露齿门控牙齿美白
        if whiten_teeth_strength > 0.0 and target.face is not None:
            mouth = polygon_mask(target.face[LIPS_INNER], (h, w))
            check = check_teeth_eligibility(original_f, target.face, mouth, target.face_width())
            if not check['eligible']:
                skipped.append({
                    'tool': 'whiten_teeth',
                    'person': target.id,
                    'reason': '未确认到可处理的露齿区域',
                    'check': check,
                })
                warnings.append('未确认到可处理的露齿区域，牙齿美白已跳过。')
            else:
                img_f = whiten_teeth(img_f, mouth, float(whiten_teeth_strength))
                applied.append(f'whiten-teeth {whiten_teeth_strength}')

        # 4. 引导细化无光晕背景虚化
        if blur_bg > 0.0:
            img_f = blur_background(img_f, target.mask, float(blur_bg))
            applied.append(f'blur-bg {blur_bg}')

        img = to_uint8(img_f)

        # 5. 合并几何位移场（瘦脸 + 旋转椭圆大眼），单次重采样
        field = new_displacement_field((h, w))
        if slim_face > 0.0 and target.face is not None:
            add_slim_face(field, target.face, float(slim_face))
            applied.append(f'slim-face {slim_face}')
        if enlarge_eyes > 0.0 and target.face is not None:
            add_enlarge_eyes(field, target.face, float(enlarge_eyes))
            applied.append(f'enlarge-eyes {enlarge_eyes}')
        if np.any(field):
            img, safe_scale = apply_displacement(img, field)
            if safe_scale < 0.999:
                warnings.append(f'几何变形强度已自动按 {round(safe_scale * 100)}% 比例防撕裂平滑收敛。')

        out_p = Path(output_path)
        write_image_bgr(out_p, img, quality=quality)

        total_seconds = round(time.perf_counter() - t0, 3)
        result = {
            'engine': 'lumina-native-v1',
            'input': str(p),
            'output': str(out_p),
            'size': [w, h],
            'people_detected': len(scene.people),
            'targets': [target.id],
            'applied': applied,
            'warnings': warnings,
            'skipped': skipped,
        }
        meta = {
            'engine': 'lumina-native-v1',
            'in_process': True,
            'cache_hit': cache_hit,
            'detect_ms': detect_ms,
            'seconds': total_seconds,
            'exit_code': 0,
        }
        return result, meta


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_ENGINE: LuminaRetouchEngine | None = None


def get_default_engine() -> LuminaRetouchEngine:
    global _DEFAULT_ENGINE
    with _DEFAULT_LOCK:
        if _DEFAULT_ENGINE is None:
            _DEFAULT_ENGINE = LuminaRetouchEngine()
        return _DEFAULT_ENGINE


def _cli() -> None:
    parser = argparse.ArgumentParser(description='Lumina 原生六工具人像精修引擎 CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_an = sub.add_parser('analyze', help='分析人像关键点、肤质与露齿资格')
    p_an.add_argument('image', help='输入人像路径')
    p_an.add_argument('--overlay', default=None, help='可选可视化叠加图输出路径')

    p_ed = sub.add_parser('edit', help='按参数从原图执行人像精修')
    p_ed.add_argument('image', help='输入原图路径')
    p_ed.add_argument('-o', '--output', required=True, help='输出图像路径')
    p_ed.add_argument('--person', type=int, default=0, help='目标人物索引（默认 0）')
    p_ed.add_argument('--slim-face', type=float, default=0.0)
    p_ed.add_argument('--enlarge-eyes', type=float, default=0.0)
    p_ed.add_argument('--remove-acne', type=float, default=None)
    p_ed.add_argument('--smooth-skin', type=float, default=0.0)
    p_ed.add_argument('--blur-bg', type=float, default=0.0)
    p_ed.add_argument('--whiten-teeth', type=float, default=0.0)

    args = parser.parse_args()
    engine = get_default_engine()
    if args.cmd == 'analyze':
        res, meta = engine.analyze(args.image, overlay_path=args.overlay)
    else:
        res, meta = engine.edit(
            args.image,
            args.output,
            slim_face=args.slim_face,
            enlarge_eyes=args.enlarge_eyes,
            remove_acne=args.remove_acne,
            smooth_skin_strength=args.smooth_skin,
            blur_bg=args.blur_bg,
            whiten_teeth_strength=args.whiten_teeth,
            person_index=args.person,
        )
    print(json.dumps({'result': res, 'meta': meta}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    _cli()
