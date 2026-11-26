"""Lumina 原生几何微调算子：连续轮廓瘦脸 + 旋转椭圆自适应大眼。

设计要点：
1. 所有几何变形累加至单一反向位移场 (dx, dy)，整图仅执行一次三次样条重采样，避免多次插值模糊。
2. 瘦脸采用 6 对渐进式面颊-下颌锚点 + 下巴/双耳/嘴角三重保护场，收紧轮廓同时不拉歪五官。
3. 大眼沿每只眼睛的内外眼角主轴构造旋转椭圆放大场，并对眉毛区域做柔和衰减保护，避免圆球凸眼感。
4. 内置雅可比行列式防折叠约束，高强度叠加也能保证零撕裂。
"""
from __future__ import annotations

import cv2
import numpy as np

from .analyzer import (
    BROW_L,
    BROW_R,
    FACE_CHIN,
    FACE_LEFT_EDGE,
    FACE_RIGHT_EDGE,
    FACE_TOP,
    IRIS_L,
    IRIS_R,
    LIPS_OUTER,
)

# 6 对渐进式面颊/下颌轮廓点 (人物右侧, 人物左侧, 相对权重)
JAW_CONTOUR_PAIRS = [
    (93, 323, 0.22),    # 颧骨下缘轻收
    (132, 361, 0.45),   # 上下颌过渡
    (58, 288, 1.00),    # 咬肌/下颌角主收紧区
    (172, 397, 0.90),   # 下颌中段
    (136, 365, 0.52),   # 下颌前段
    (150, 379, 0.24),   # 近下巴柔和收尾
]
EAR_ANCHORS = ((234, 93), (454, 323))
EYE_SPECS = [
    # (虹膜中心, 外眼角, 内眼角, 上眼睑中点, 下眼睑中点, 对应眉毛点集)
    (IRIS_R, 33, 133, 159, 145, BROW_R),
    (IRIS_L, 263, 362, 386, 374, BROW_L),
]

FACE_SLIM_PULL = 0.036
FACE_SLIM_RADIUS = 0.23
CHIN_GUARD_SIGMA = 0.165
MOUTH_GUARD_SIGMA = 0.145
EAR_GUARD_SIGMA = (0.095, 0.165)
EYE_SCALE_MAX = 0.26
MIN_JACOBIAN = 0.28


def new_displacement_field(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros((*shape, 2), dtype=np.float32)


def _roi_grid(shape: tuple[int, int], cx: float, cy: float, rx: float, ry: float):
    h, w = shape
    x0 = max(0, int(np.floor(cx - rx)))
    x1 = min(w, int(np.ceil(cx + rx)) + 1)
    y0 = max(0, int(np.floor(cy - ry)))
    y1 = min(h, int(np.ceil(cy + ry)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    return y0, y1, x0, x1, gx, gy


def _add_smooth_push(
    field: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
) -> None:
    move = (end - start).astype(np.float32)
    dist = float(np.linalg.norm(move))
    if dist < 1e-3 or radius < 2.0:
        return
    if dist > radius * 0.48:
        move *= (radius * 0.48) / dist
    roi = _roi_grid(field.shape[:2], float(start[0]), float(start[1]), radius, radius)
    if roi is None:
        return
    y0, y1, x0, x1, gx, gy = roi
    r2 = ((gx - start[0]) ** 2 + (gy - start[1]) ** 2) / (radius * radius)
    weight = np.maximum(1.0 - r2, 0.0) ** 2
    field[y0:y1, x0:x1, 0] -= weight * move[0]
    field[y0:y1, x0:x1, 1] -= weight * move[1]


def _slim_protection_mask(
    shape: tuple[int, int],
    face: np.ndarray,
    width: float,
    height: float,
    top: np.ndarray,
    chin: np.ndarray,
    axis: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """在面部 ROI 内计算下巴、嘴部与双耳的保护权重（0=完全保护，1=允许微调）。"""
    y0, y1, x0, x1 = bbox
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

    # 下巴保护（避免尖下巴畸变）
    guard = 1.0 - np.exp(
        -((gx - chin[0]) ** 2 + (gy - chin[1]) ** 2) / (2.0 * (CHIN_GUARD_SIGMA * width) ** 2)
    )

    # 嘴部保护（避免嘴角被向内拉歪）
    mouth_center = face[LIPS_OUTER].mean(axis=0)
    mouth_sx = max(4.0, float(np.ptp(face[LIPS_OUTER, 0])) * 0.72)
    mouth_sy = max(4.0, MOUTH_GUARD_SIGMA * width)
    guard *= 1.0 - 0.85 * np.exp(
        -(((gx - mouth_center[0]) / mouth_sx) ** 2 + ((gy - mouth_center[1]) / mouth_sy) ** 2) / 2.0
    )

    # 双耳保护（避免耳朵被拉进脸廓）
    sx, sy = EAR_GUARD_SIGMA[0] * width, EAR_GUARD_SIGMA[1] * height
    for eye_level, lobe_level in EAR_ANCHORS:
        edge = (face[eye_level] + face[lobe_level]) * 0.5
        on_axis = top + axis * float((edge - top) @ axis)
        outward = (edge - on_axis) / (np.linalg.norm(edge - on_axis) + 1e-6)
        ear = edge + outward * 0.08 * width
        guard *= 1.0 - np.exp(-(((gx - ear[0]) / sx) ** 2 + ((gy - ear[1]) / sy) ** 2) / 2.0)

    return np.clip(guard, 0.0, 1.0).astype(np.float32)


def add_slim_face(field: np.ndarray, face: np.ndarray, strength: float) -> None:
    """沿面部中轴对称收紧下颌与面颊轮廓。"""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-4:
        return
    h, w = field.shape[:2]
    width = float(np.linalg.norm(face[FACE_LEFT_EDGE] - face[FACE_RIGHT_EDGE]))
    top, chin = face[FACE_TOP], face[FACE_CHIN]
    height = float(np.linalg.norm(chin - top))
    if width < 16.0 or height < 16.0:
        return

    axis = (chin - top) / (height + 1e-6)
    radius = width * FACE_SLIM_RADIUS

    # 仅在面部外扩 ROI 内累加，避免全图扫描开销
    fx0, fy0 = face.min(axis=0)
    fx1, fy1 = face.max(axis=0)
    pad = int(np.ceil(radius * 1.15))
    x0, x1 = max(0, int(fx0) - pad), min(w, int(fx1) + pad)
    y0, y1 = max(0, int(fy0) - pad), min(h, int(fy1) + pad)
    if x0 >= x1 or y0 >= y1:
        return

    local_field = np.zeros((y1 - y0, x1 - x0, 2), dtype=np.float32)
    origin = np.array([x0, y0], dtype=np.float32)

    for right_id, left_id, weight in JAW_CONTOUR_PAIRS:
        for pid in (right_id, left_id):
            p = face[pid]
            on_axis = top + axis * float((p - top) @ axis)
            inward = (on_axis - p) / (np.linalg.norm(on_axis - p) + 1e-6)
            delta = inward * (width * FACE_SLIM_PULL * s * weight)
            _add_smooth_push(local_field, p - origin, p + delta - origin, radius)

    guard = _slim_protection_mask((h, w), face, width, height, top, chin, axis, (y0, y1, x0, x1))
    field[y0:y1, x0:x1] += local_field * guard[..., None]


def add_enlarge_eyes(field: np.ndarray, face: np.ndarray, strength: float) -> None:
    """旋转椭圆自适应大眼：顺应眼型倾斜角，同时保护眉弓形态。"""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-4:
        return
    h, w = field.shape[:2]

    for iris_id, outer_id, inner_id, upper_id, lower_id, brow_ids in EYE_SPECS:
        center = face[iris_id]
        vec = face[inner_id] - face[outer_id]
        eye_w = float(np.linalg.norm(vec))
        eye_h = float(np.linalg.norm(face[upper_id] - face[lower_id]))
        if eye_w < 6.0:
            continue

        # 局部眼轴单位正交基 (u: 眼长方向, v: 眼高方向)
        ux, uy = float(vec[0] / eye_w), float(vec[1] / eye_w)
        vx, vy = -uy, ux

        rx = eye_w * 1.05
        ry = max(eye_h * 2.35, eye_w * 0.78)
        bound = max(rx, ry) * 1.15
        roi = _roi_grid((h, w), float(center[0]), float(center[1]), bound, bound)
        if roi is None:
            continue

        y0, y1, x0, x1, gx, gy = roi
        dx = gx - center[0]
        dy = gy - center[1]

        # 投影到旋转椭圆局部坐标系
        du = dx * ux + dy * uy
        dv = dx * vx + dy * vy
        r2 = (du / rx) ** 2 + (dv / ry) ** 2
        falloff = np.maximum(1.0 - r2, 0.0) ** 2

        # 保护眉毛区域不被过度向上顶起
        brow_center = face[brow_ids].mean(axis=0)
        brow_dist_y = max(eye_h * 1.2, float(np.linalg.norm(brow_center - center)))
        brow_sx = eye_w * 0.95
        brow_sy = brow_dist_y * 0.45
        brow_guard = 1.0 - 0.65 * np.exp(
            -(((gx - brow_center[0]) / brow_sx) ** 2 + ((gy - brow_center[1]) / brow_sy) ** 2) / 2.0
        )

        # 纵向放大略高于横向，眼神更清亮自然
        amount_u = EYE_SCALE_MAX * 0.88 * s
        amount_v = EYE_SCALE_MAX * 1.08 * s
        shift_u = -du * amount_u * falloff * brow_guard
        shift_v = -dv * amount_v * falloff * brow_guard

        field[y0:y1, x0:x1, 0] += shift_u * ux + shift_v * vx
        field[y0:y1, x0:x1, 1] += shift_u * uy + shift_v * vy


def limit_folding(field: np.ndarray) -> float:
    """检查位移场雅可比行列式，防止强变形导致像素折叠或撕裂。"""
    scale = 1.0
    for _ in range(18):
        dx_dy, dx_dx = np.gradient(field[..., 0])
        dy_dy, dy_dx = np.gradient(field[..., 1])
        det = (1.0 + dx_dx) * (1.0 + dy_dy) - dx_dy * dy_dx
        if float(det.min()) >= MIN_JACOBIAN:
            break
        field *= 0.85
        scale *= 0.85
    return scale


def apply_displacement(img: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, float]:
    """应用累计位移场，返回 (重采样后图像, 雅可比安全缩放因子)。"""
    if not np.any(field):
        return img, 1.0
    scale = limit_folding(field)
    h, w = img.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(
        img,
        gx + field[..., 0],
        gy + field[..., 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped, scale
