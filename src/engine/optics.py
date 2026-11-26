"""Lumina 原生光学与局部精修算子：无光晕散景背景虚化 + 露齿门控自然牙齿美白。

关键优化：
1. 背景虚化（blur_background）：
   - 利用原图灰度对 256×256 人像掩码做保边引导细化（Guided Alpha Refinement），
     发丝与肩线轮廓大幅贴合真实边缘。
   - 使用归一化卷积散景模糊，完全避免前景颜色溢出到虚化背景形成“白边/灰晕”。
2. 牙齿美白（whiten_teeth）：
   - 保持严格的露齿资格门控（开口比、嘴内高饱和异物遮挡筛查、有效亮区占比），闭嘴自动跳过。
   - Lab 空间精准去黄（压降 b*）+ 温和提亮（L*）+ 齿缝立体度保留，杜绝荧光假白。
"""
from __future__ import annotations

import cv2
import numpy as np

from .skin import feather_mask, to_uint8


def _guided_refine_alpha(guide_gray: np.ndarray, alpha: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """用高分辨率原图灰度引导细化粗分割 Alpha 掩码，贴合发丝与边缘。"""
    size = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide_gray, -1, size)
    mean_p = cv2.boxFilter(alpha, -1, size)
    corr_i = cv2.boxFilter(guide_gray * guide_gray, -1, size)
    corr_ip = cv2.boxFilter(guide_gray * alpha, -1, size)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, -1, size)
    mean_b = cv2.boxFilter(b, -1, size)
    return np.clip(mean_a * guide_gray + mean_b, 0.0, 1.0)


def blur_background(img_f: np.ndarray, subject_mask: np.ndarray, strength: float) -> np.ndarray:
    """保边细化人像散景虚化（无溢色光晕）。"""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-4:
        return img_f

    h, w = img_f.shape[:2]
    diag = float(np.hypot(h, w))
    sigma = max(1.2, s * 0.026 * diag)

    # 1. 原图引导细化前景 Alpha，让发丝和肩部边缘自然贴合
    guide = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
    refine_r = max(4, min(14, int(round(diag * 0.006))))
    refined = _guided_refine_alpha(guide, np.clip(subject_mask, 0.0, 1.0), radius=refine_r, eps=2e-3)
    # 稍微增强前景主体置信度对比，避免主体内部误模糊
    keep = np.clip((refined - 0.12) / 0.76, 0.0, 1.0)
    keep = feather_mask(keep, max(0.8, diag * 0.0012))

    # 2. 对背景权重略微收缩前景边界，防止主体颜色渗入背景散景
    erode_px = max(3, int(round(diag * 0.003))) | 1
    bg_raw = np.clip(1.0 - keep, 0.0, 1.0)
    bg_weight = cv2.erode(bg_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px)))

    # 3. 归一化卷积散景模糊（大图自动半分辨率加速，保持高质量与毫秒级响应）
    if sigma > 10.0 and max(h, w) > 960:
        scale = 0.5
        sw, sh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
        img_s = cv2.resize(img_f, (sw, sh), interpolation=cv2.INTER_AREA)
        bg_s = cv2.resize(bg_weight, (sw, sh), interpolation=cv2.INTER_AREA)
        sig_s = sigma * scale
        num_s = cv2.GaussianBlur(img_s * bg_s[..., None], (0, 0), sig_s)
        den_s = cv2.GaussianBlur(bg_s, (0, 0), sig_s)[..., None]
        blur_s = np.where(den_s > 1e-3, num_s / np.maximum(den_s, 1e-3), img_s)
        blurred = cv2.resize(blur_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        num = cv2.GaussianBlur(img_f * bg_weight[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(bg_weight, (0, 0), sigma)[..., None]
        blurred = np.where(den > 1e-3, num / np.maximum(den, 1e-3), img_f)

    return blurred + (img_f - blurred) * keep[..., None]


def check_teeth_eligibility(
    img_f: np.ndarray,
    face: np.ndarray,
    inner_mouth_mask: np.ndarray,
    face_width: float,
) -> dict:
    """保守的露齿可见性门控（防闭嘴误处理与彩色异物遮挡）。"""
    opening = float(np.linalg.norm(face[13] - face[14]) / max(face_width, 1.0))
    area = float(inner_mouth_mask.sum())
    if area < 20.0:
        return {
            'eligible': False,
            'opening_ratio': round(opening, 4),
            'candidate_fraction': 0.0,
            'candidate_area_ratio': 0.0,
            'colored_mouth_fraction': 0.0,
            'suspected_occlusion': False,
        }

    hsv = cv2.cvtColor(to_uint8(img_f), cv2.COLOR_BGR2HSV)
    mouth_px = inner_mouth_mask > 0.5
    colored = (hsv[..., 1] > 110) & (hsv[..., 2] > 90)
    colored_fraction = float(colored[mouth_px].mean()) if mouth_px.any() else 0.0
    suspected_occlusion = colored_fraction > 0.30

    lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)
    candidate = (
        inner_mouth_mask
        * np.clip((lab[..., 0] - 40.0) / 15.0, 0.0, 1.0)
        * np.clip((18.0 - lab[..., 1]) / 8.0, 0.0, 1.0)
    )
    pixels = int((candidate > 0.35).sum())
    fraction = pixels / area
    relative_area = pixels / max(face_width ** 2, 1.0)
    eligible = bool(
        not suspected_occlusion
        and opening >= 0.02
        and pixels >= 20
        and fraction >= 0.10
        and relative_area >= 0.0015
    )
    return {
        'eligible': eligible,
        'opening_ratio': round(opening, 4),
        'candidate_fraction': round(fraction, 4),
        'candidate_area_ratio': round(relative_area, 4),
        'colored_mouth_fraction': round(colored_fraction, 4),
        'suspected_occlusion': suspected_occlusion,
    }


def whiten_teeth(img_f: np.ndarray, inner_mouth_mask: np.ndarray, strength: float) -> np.ndarray:
    """自然牙齿净白：仅针对嘴内亮区去黄提亮，保护唇缘、牙龈与自然齿缝层次。"""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 1e-4 or float(inner_mouth_mask.sum()) < 20.0:
        return img_f

    lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)
    l_ch = lab[..., 0].copy()
    a_ch = lab[..., 1]
    b_ch = lab[..., 2]

    # 高亮度 + 低红度（排除牙龈、舌头、内唇）
    teeth_mask = (
        inner_mouth_mask
        * np.clip((l_ch - 42.0) / 16.0, 0.0, 1.0)
        * np.clip((17.0 - a_ch) / 8.0, 0.0, 1.0)
    )
    teeth_mask = feather_mask(teeth_mask, 1.0)

    # 压降黄色分量 b*，克制提亮 L*（防止失真惨白）
    lab[..., 2] = b_ch - np.maximum(b_ch, 0.0) * (0.68 * s)
    lab[..., 1] = a_ch - np.maximum(a_ch, 0.0) * (0.28 * s)
    # 中高亮区温和提亮，暗缝保留阴影层次
    midtone_lift = np.clip((l_ch - 35.0) / 55.0, 0.0, 1.0)
    lab[..., 0] = np.clip(l_ch + (96.0 - l_ch) * (0.16 * s) * midtone_lift, 0.0, 100.0)

    edited = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0.0, 1.0)
    return img_f + (edited - img_f) * teeth_mask[..., None]
