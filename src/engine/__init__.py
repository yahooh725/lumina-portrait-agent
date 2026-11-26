"""Lumina 原生人像精修引擎。

模块划分：
- analyzer.py : 人脸 478 关键点与多类人像分割（持久化模型实例 + 按图 SHA256 缓存）
- geometry.py : 瘦脸（连续下颌轮廓牵引 + 嘴角/下巴/双耳保护）与旋转椭圆自适应大眼
- skin.py     : Lab 高低频磨皮（保毛孔与光影立体感）与防伪影自适应祛痘
- optics.py   : 引导细化无光晕背景虚化与露齿门控自然牙齿美白
- pipeline.py : 进程内统一渲染管线 `LuminaRetouchEngine` 与 CLI 入口
"""
from .pipeline import LuminaRetouchEngine, get_default_engine

__all__ = ['LuminaRetouchEngine', 'get_default_engine']
