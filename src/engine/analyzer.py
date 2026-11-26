"""Lumina 原生人像解析器：MediaPipe 478 点人脸网格 + 多类人像语义分割。

核心工程优化：
1. 持久化加载模型实例（进程内仅初始化一次，消除重复子进程冷启动）。
2. 按原图 SHA256 缓存解析结果（LRU），多轮连续修图无需重复跑检测，渲染提速数十倍。
3. 单人人像场景跳过冗余的全身骨架模型，仅按需加载 FaceLandmarker 与 ImageSegmenter。
"""
from __future__ import annotations

import atexit
from collections import OrderedDict
import contextlib
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import threading
import urllib.request

import cv2
import numpy as np

os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / 'tool_acceptance/models'

MODEL_URLS = {
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
        'face_landmarker/float16/1/face_landmarker.task'
    ),
    'selfie_multiclass_256x256.tflite': (
        'https://storage.googleapis.com/mediapipe-models/image_segmenter/'
        'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
    ),
}

# Selfie-multiclass 分割类别索引
SEG_BACKGROUND, SEG_HAIR, SEG_BODY_SKIN, SEG_FACE_SKIN, SEG_CLOTHES, SEG_OTHERS = range(6)

# MediaPipe 478 点拓扑关键索引
FACE_LEFT_EDGE, FACE_RIGHT_EDGE, FACE_TOP, FACE_CHIN = 234, 454, 10, 152
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
BROW_L = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSE_ALAR_R, NOSE_ALAR_L, NOSE_TIP, SUBNASALE = 98, 327, 1, 2
IRIS_R, IRIS_L = 468, 473
FOREHEAD_CONTOUR = [10, 338, 297, 332, 284, 251, 389, 162, 21, 54, 103, 67, 109]
FOREHEAD_EXTEND = 0.24


def resolve_model_dir() -> Path:
    env = os.environ.get('LUMINA_VISION_MODEL_DIR') or os.environ.get('PHOTO_RETOUCH_MODEL_DIR')
    if env:
        return Path(env).expanduser()
    if DEFAULT_MODEL_DIR.is_dir():
        return DEFAULT_MODEL_DIR
    return Path.home() / '.cache/lumina-portrait/models'


def ensure_model_file(name: str) -> str:
    model_dir = resolve_model_dir()
    path = model_dir / name
    if not path.is_file():
        model_dir.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(MODEL_URLS[name], part)
        part.rename(path)
    return str(path)


@contextlib.contextmanager
def quiet_native_stderr():
    """屏蔽 MediaPipe C++/TFLite 初始化冗余日志。"""
    if os.environ.get('LUMINA_ENGINE_VERBOSE') == '1':
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def read_image_bgr(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'找不到图像文件：{p}')
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'无法解码图像文件：{p}')
    return img


def write_image_bgr(path: str | Path, img: np.ndarray, quality: int = 95) -> None:
    p = Path(path)
    ext = p.suffix.lower() or '.png'
    if ext in ('.jpg', '.jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.webp':
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    else:
        params = []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f'无法编码输出图像格式：{ext}')
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(p))


def polygon_mask(points: np.ndarray, shape: tuple[int, int], convex: bool = False) -> np.ndarray:
    mask = np.zeros(shape, np.float32)
    pts = np.round(points).astype(np.int32)
    if len(pts) < 3:
        return mask
    cv2.fillPoly(mask, [cv2.convexHull(pts) if convex else pts], 1.0)
    return mask


def feature_protect_mask(face: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """五官保护掩码：眼睛、眉毛、嘴唇及鼻翼褶皱，避免磨皮或祛痘误伤清晰结构。"""
    width = float(np.ptp(face[:, 0]))
    mask = np.zeros(shape, np.float32)
    groups = (
        (EYE_R, True, 0.058),
        (EYE_L, True, 0.058),
        (BROW_R, True, 0.028),
        (BROW_L, True, 0.028),
        (LIPS_OUTER, False, 0.026),
    )
    for idxs, convex, margin in groups:
        pad = max(3, int(width * margin)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad))
        mask = np.maximum(mask, cv2.dilate(polygon_mask(face[idxs], shape, convex), kernel))

    alar = float(np.linalg.norm(face[NOSE_ALAR_R] - face[NOSE_ALAR_L]))
    center = face[[NOSE_ALAR_R, NOSE_ALAR_L, SUBNASALE, NOSE_TIP]].mean(0)
    nose = np.zeros(shape, np.float32)
    cv2.ellipse(
        nose,
        (int(round(center[0])), int(round(center[1]))),
        (max(2, int(0.68 * alar)), max(2, int(0.48 * alar))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    pad = max(3, int(width * 0.024)) | 1
    nose = cv2.dilate(nose, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad, pad)))
    return np.clip(np.maximum(mask, nose), 0.0, 1.0)


def face_outline_mask(face: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """覆盖发际线至下颌的完整面部轮廓区域（向外补充额头高度）。"""
    height = float(np.linalg.norm(face[FACE_TOP] - face[FACE_CHIN]))
    up = (face[FACE_TOP] - face[FACE_CHIN]) / (height + 1e-6)
    forehead_ext = face[FOREHEAD_CONTOUR] + up * (FOREHEAD_EXTEND * height)
    pts = np.vstack([face, forehead_ext])
    return polygon_mask(pts, shape, convex=True)


def _as_float_mask(mp_mask, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mp_mask.numpy_view(), dtype=np.float32).squeeze()
    if arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(arr, 0.0, 1.0)


def _mask_bbox(mask: np.ndarray, fallback: np.ndarray | None = None) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs):
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    if fallback is not None and len(fallback):
        x0, y0 = fallback.min(0)
        x1, y1 = fallback.max(0)
        return int(x0), int(y0), int(x1), int(y1)
    return 0, 0, 0, 0


@dataclass
class DetectedPerson:
    id: int
    face: np.ndarray | None
    bbox: tuple[int, int, int, int]
    mask: np.ndarray

    def face_width(self) -> float:
        if self.face is None:
            return 0.0
        return float(np.linalg.norm(self.face[FACE_LEFT_EDGE] - self.face[FACE_RIGHT_EDGE]))


@dataclass
class PortraitScene:
    image_sha256: str
    shape: tuple[int, int]
    people: list[DetectedPerson]
    foreground: np.ndarray
    background: np.ndarray
    skin: np.ndarray
    hair: np.ndarray
    clothes: np.ndarray

    def person_skin(self, person: DetectedPerson) -> np.ndarray:
        """提取目标人物皮肤掩码（综合语义皮肤分割与面部轮廓，剔除发丝与衣物）。"""
        skin = self.skin * person.mask
        if person.face is not None:
            outline = face_outline_mask(person.face, self.shape)
            non_hair_clothes = np.clip(1.0 - np.maximum(self.hair, self.clothes), 0.0, 1.0)
            skin = np.maximum(skin, outline * non_hair_clothes * 0.92)
        return np.clip(skin, 0.0, 1.0)

    def face_skin_for_blemish(self, person: DetectedPerson) -> np.ndarray:
        """用于祛痘的精细面部皮肤掩码（严格剔除五官、发丝与背景边缘）。"""
        if person.face is None:
            return np.zeros(self.shape, np.float32)
        outline = face_outline_mask(person.face, self.shape)
        base_skin = self.person_skin(person) * outline
        # 向内收缩 2~4px，避免将面部轮廓边缘阴影误判为痘痘
        erode_k = max(3, int(person.face_width() * 0.015)) | 1
        inner = cv2.erode(
            (base_skin > 0.45).astype(np.float32),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)),
        )
        protect = feature_protect_mask(person.face, self.shape)
        return np.clip(base_skin * inner * (1.0 - protect), 0.0, 1.0)

    def retouch_skin_mask(self, person: DetectedPerson) -> np.ndarray:
        """用于磨皮的羽化皮肤掩码（保留五官清晰度）。"""
        skin = self.person_skin(person)
        if person.face is not None:
            protect = feature_protect_mask(person.face, self.shape)
            protect_soft = cv2.GaussianBlur(protect, (0, 0), max(1.2, person.face_width() * 0.008))
            skin = skin * (1.0 - protect_soft)
        return np.clip(skin, 0.0, 1.0)


class PortraitAnalyzer:
    """持久化 MediaPipe 解析器 + LRU 缓存。"""

    def __init__(self, cache_size: int = 16, max_faces: int = 4):
        self._cache_size = max(1, cache_size)
        self._max_faces = max_faces
        self._cache: OrderedDict[str, PortraitScene] = OrderedDict()
        self._lock = threading.Lock()
        self._mp = None
        self._face_landmarker = None
        self._segmenter = None
        atexit.register(self.close)

    def close(self) -> None:
        with self._lock:
            for attr in ('_face_landmarker', '_segmenter'):
                obj = getattr(self, attr, None)
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass
                    setattr(self, attr, None)

    def _ensure_models(self) -> None:
        if self._face_landmarker is not None and self._segmenter is not None:
            return
        face_path = ensure_model_file('face_landmarker.task')
        seg_path = ensure_model_file('selfie_multiclass_256x256.tflite')
        with quiet_native_stderr():
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            image_mode = vision.RunningMode.IMAGE
            self._face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=face_path),
                    running_mode=image_mode,
                    num_faces=self._max_faces,
                )
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=seg_path),
                    running_mode=image_mode,
                    output_confidence_masks=True,
                    output_category_mask=False,
                )
            )
            self._mp = mp

    def analyze_bgr(self, img: np.ndarray, image_sha256: str | None = None) -> tuple[PortraitScene, bool]:
        """解析 BGR 图像，返回 (PortraitScene, cache_hit)。"""
        key = image_sha256 or hashlib.sha256(img.tobytes()).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached.shape == img.shape[:2]:
                self._cache.move_to_end(key)
                return cached, True

            self._ensure_models()
            h, w = img.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

            with quiet_native_stderr():
                face_res = self._face_landmarker.detect(mp_img)
                seg_res = self._segmenter.segment(mp_img)

            classes = [_as_float_mask(m, (h, w)) for m in seg_res.confidence_masks]
            background = classes[SEG_BACKGROUND]
            foreground = np.clip(1.0 - background, 0.0, 1.0)
            skin = np.clip(classes[SEG_BODY_SKIN] + classes[SEG_FACE_SKIN], 0.0, 1.0)
            hair = classes[SEG_HAIR]
            clothes = classes[SEG_CLOTHES]

            faces = [
                np.array([[lm.x * w, lm.y * h] for lm in pts], dtype=np.float32)
                for pts in (face_res.face_landmarks or [])
            ]
            faces.sort(key=lambda pts: float(pts[:, 0].mean()))

            people: list[DetectedPerson] = []
            for idx, face in enumerate(faces):
                bbox = _mask_bbox(foreground if len(faces) == 1 else face_outline_mask(face, (h, w)), face)
                people.append(DetectedPerson(id=idx, face=face, bbox=bbox, mask=foreground))

            scene = PortraitScene(
                image_sha256=key,
                shape=(h, w),
                people=people,
                foreground=foreground,
                background=background,
                skin=skin,
                hair=hair,
                clothes=clothes,
            )
            self._cache[key] = scene
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return scene, False

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
