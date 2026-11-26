"""Lumina 原生肤质精修算子：防伪影自适应祛痘 + Lab 双层高低频磨皮。

针对旧版算子的关键改进：
1. 祛痘（remove_acne）：
   - 解决高 sensitivity 时把正常肤质误判成上百个圆形补丁的问题。
   - 引入稳健非线性阈值曲线 + Lab 色差绝对下限 + 显著度排序与合理数量上限。
   - 在每颗瑕疵的局部 ROI 上使用 Telea 纹理修补 + 局部皮肤低频混合 + 高频微纹理回填，
     彻底消除生硬的圆形“贴片感”。
2. 磨皮（smooth_skin）：
   - 在 CIE Lab 空间分离高频毛孔细节与低频肤色/光影，亮度和色度通道分别平滑，
     保留鼻梁、下颌与苹果肌的立体光影，避免塑料假面感。
"""
from __future__ import annotations

import cv2
import numpy as np

BLEMISH_MEDIAN_RATIO = 0.055
BLEMISH_MIN_R_RATIO = 0.0045
BLEMISH_MAX_R_RATIO = 0.032
WHITEHEAD_MAX_R_RATIO = 0.011
BLEMISH_MIN_PX = 1.2
NOISE_FLOOR_LAB = 1.8
JPEG_DENOISE_SIGMA = 0.85

SMOOTH_RADIUS_RATIO = 0.024
SMOOTH_EPS_L = 0.028 ** 2
SMOOTH_EPS_AB = 0.045 ** 2


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def _odd(val: float, lo: int = 3) -> int:
    return max(lo, int(round(val))) | 1


def feather_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.1:
        return mask
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)


def guided_filter_single(src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    size = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(src, -1, size)
    var_i = cv2.boxFilter(src * src, -1, size) - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    return cv2.boxFilter(a, -1, size) * src + cv2.boxFilter(b, -1, size)


def find_blemishes(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    face_width: float,
    sensitivity: float = 0.5,
) -> np.ndarray:
    """检测面部局部炎性红痘、暗沉痘印与小型白头，返回按显著度排序的 (N, 3) [x, y, r]。"""
    s = float(np.clip(sensitivity, 0.0, 1.0))
    inside = skin_mask > 0.5
    if inside.sum() < 120 or face_width < 40.0:
        return np.zeros((0, 3), dtype=np.float32)

    k = _odd(face_width * BLEMISH_MEDIAN_RATIO, 5)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_ch = cv2.GaussianBlur(np.ascontiguousarray(lab[..., 0]), (0, 0), JPEG_DENOISE_SIGMA)
    a_ch = cv2.GaussianBlur(np.ascontiguousarray(lab[..., 1]), (0, 0), JPEG_DENOISE_SIGMA)

    dark_diff = cv2.medianBlur(l_ch, k).astype(np.float32) - l_ch.astype(np.float32)
    red_diff = a_ch.astype(np.float32) - cv2.medianBlur(a_ch, k).astype(np.float32)

    def robust_z(field: np.ndarray) -> np.ndarray:
        vals = field[inside]
        med = float(np.median(vals))
        mad = max(float(np.median(np.abs(vals - med))) * 1.4826, NOISE_FLOOR_LAB)
        return (field - med) / mad

    z_red = robust_z(red_diff)
    z_dark = robust_z(dark_diff)
    z_bright = robust_z(-dark_diff)

    # 非线性阈值曲线：sensitivity=1.0 时稳定停留在 2.55，防止误检正常肤质
    thr = 4.6 - 2.05 * (s ** 0.85)
    # 绝对 Lab 色差下限：非常干净的皮肤不因 MAD 偏小产生假阳性
    min_red_abs = 2.6 - 1.1 * s
    min_dark_abs = 4.5 - 1.5 * s

    score_map = z_red + 0.55 * np.maximum(z_dark, 0.0)
    inflamed = (
        inside
        & (z_red > thr * 0.62)
        & (score_map > thr)
        & ((red_diff > min_red_abs) | (dark_diff > min_dark_abs))
    )
    whitehead = inside & (z_bright > thr * 2.05) & (-dark_diff > 6.0)

    r_min = max(BLEMISH_MIN_PX, face_width * BLEMISH_MIN_R_RATIO)
    r_max = face_width * BLEMISH_MAX_R_RATIO
    r_white = face_width * WHITEHEAD_MAX_R_RATIO

    candidates: list[tuple[float, float, float, float]] = []
    candidates.extend(_extract_scored_blobs(inflamed, score_map, r_min, r_max))
    candidates.extend(_extract_scored_blobs(whitehead, z_bright, r_min, r_white))

    if not candidates:
        return np.zeros((0, 3), dtype=np.float32)

    # 按异常显著度降序排列，并做距离去重与合理数量上限保护
    candidates.sort(key=lambda item: item[3], reverse=True)
    max_spots = int(round(28 + 36 * s))
    kept: list[tuple[float, float, float]] = []
    for x, y, r, _score in candidates:
        if any((x - kx) ** 2 + (y - ky) ** 2 < (0.8 * max(r, kr)) ** 2 for kx, ky, kr in kept):
            continue
        kept.append((x, y, r))
        if len(kept) >= max_spots:
            break
    return np.array(kept, dtype=np.float32).reshape(-1, 3)


def _extract_scored_blobs(
    hits: np.ndarray,
    score_map: np.ndarray,
    r_min: float,
    r_max: float,
) -> list[tuple[float, float, float, float]]:
    closed = cv2.morphologyEx(hits.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    found: list[tuple[float, float, float, float]] = []
    for i in range(1, num):
        x0, y0, bw, bh, area = stats[i]
        eq_r = float(np.sqrt(area / np.pi))
        box_r = max(bw, bh) * 0.5
        if eq_r < r_min or box_r > r_max:
            continue
        if max(bw, bh) > 2.3 * max(1, min(bw, bh)):
            continue
        if area < 0.42 * bw * bh:
            continue
        roi_lbl = labels[y0:y0 + bh, x0:x0 + bw] == i
        peak_score = float(score_map[y0:y0 + bh, x0:x0 + bw][roi_lbl].mean())
        found.append((float(centroids[i][0]), float(centroids[i][1]), box_r, peak_score))
    return found


def heal_blemishes(img_bgr: np.ndarray, spots: np.ndarray, face_width: float) -> np.ndarray:
    """局部 ROI 纹理保真修复：Telea 局部修补 + 环域柔和融合 + 原皮微粒回填。"""
    if len(spots) == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    # 提取全图极细肤质微粒，修复后轻度回填，杜绝塑料斑块
    base_f = to_float(img_bgr)
    fine_grain = base_f - cv2.GaussianBlur(base_f, (0, 0), max(0.9, face_width * 0.0022))

    for x, y, r in spots:
        cx, cy = int(round(x)), int(round(y))
        patch_r = max(2, int(np.ceil(r * 1.45 + 1.0)))
        pad = max(patch_r * 3, 12)
        x0, x1 = max(0, cx - pad), min(w, cx + pad + 1)
        y0, y1 = max(0, cy - pad), min(h, cy + pad + 1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue

        roi = out[y0:y1, x0:x1]
        inpaint_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.circle(inpaint_mask, (cx - x0, cy - y0), patch_r, 255, -1)

        # 用小半径 Telea 从瑕疵周边自然插值肤色
        inpainted = cv2.inpaint(roi, inpaint_mask, max(2, int(round(r * 1.2))), cv2.INPAINT_TELEA)

        # 构造平滑径向羽化掩码，确保边界零割裂
        soft = (inpaint_mask.astype(np.float32) / 255.0)
        soft = feather_mask(soft, max(1.0, r * 0.65))
        soft = np.clip(soft * 1.15, 0.0, 1.0)[..., None]

        roi_f = to_float(roi)
        heal_f = to_float(inpainted) + 0.42 * fine_grain[y0:y1, x0:x1]
        blended = roi_f + (heal_f - roi_f) * soft
        out[y0:y1, x0:x1] = to_uint8(blended)

    return out


def smooth_skin(
    img_f: np.ndarray,
    skin_mask: np.ndarray,
    face_width: float,
    strength: float,
) -> np.ndarray:
    """Lab 高低频人像磨皮：均匀低频斑驳与泛红，保留真实毛孔与立体光影。"""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-4 or float(skin_mask.max()) <= 1e-4:
        return img_f

    radius = max(3, int(round(face_width * SMOOTH_RADIUS_RATIO)))
    lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)

    # 归一化 Lab 到 [0, 1] 方便保边滤波参数尺度统一
    l_norm = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
    a_norm = (lab[..., 1] + 128.0) / 255.0
    b_norm = (lab[..., 2] + 128.0) / 255.0

    # 高频微纹理（毛孔、细绒毛）
    pore_sigma = max(0.9, face_width * 0.0024)
    l_fine = l_norm - cv2.GaussianBlur(l_norm, (0, 0), pore_sigma)

    # 低频保边平滑：L 通道克制平滑保护光影，a/b 通道适度均匀肤色
    l_smooth = guided_filter_single(l_norm, radius, SMOOTH_EPS_L * (0.8 + 0.8 * s))
    a_smooth = guided_filter_single(a_norm, max(2, int(radius * 1.2)), SMOOTH_EPS_AB)
    b_smooth = guided_filter_single(b_norm, max(2, int(radius * 1.2)), SMOOTH_EPS_AB)

    # 自适应纹理保留比例：轻度磨皮保留 ~55% 纹理，重度保留 ~26%
    keep_texture = 0.56 - 0.30 * s
    l_out = np.clip(l_smooth + keep_texture * l_fine, 0.0, 1.0)

    lab_out = lab.copy()
    lab_out[..., 0] = l_out * 100.0
    lab_out[..., 1] = a_smooth * 255.0 - 128.0
    lab_out[..., 2] = b_smooth * 255.0 - 128.0
    edited = np.clip(cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR), 0.0, 1.0)

    weight = np.clip(skin_mask * (s ** 0.92), 0.0, 1.0)[..., None]
    return img_f + (edited - img_f) * weight
