"""videofp —— 基于感知指纹（pHash）的视频盗版/抄袭鉴定系统。

管线::

    视频 -> 解码(容错降级) -> 黑边裁剪归一化 -> pHash64 + 镜像哈希
         -> 序列对齐 + 片段定位(含倍速搜索)
         -> 音频地标指纹多尺度匹配
         -> 融合打分 + 时间码证据

硬约束（Agent.md §2）：判定只来自算法输出数字，不使用任何视觉/模型"看图判断"。
"""

from __future__ import annotations

__version__ = "0.2.0"

from .align import AlignmentResult, MatchSegment, align_pair, segment_iou
from .decode import decode_gray, ffmpeg_path, ffprobe_path, probe
from .hashes import dct256, dhash64, hamming64, hamming_matrix, phash64, phash64_mirrored
from .index import Config, QueryResult, VideoFingerprint, build_db, load_db, query

__all__ = [
    "__version__",
    "Config",
    "align_pair",
    "AlignmentResult",
    "MatchSegment",
    "segment_iou",
    "decode_gray",
    "probe",
    "ffmpeg_path",
    "ffprobe_path",
    "phash64",
    "phash64_mirrored",
    "dct256",
    "dhash64",
    "hamming64",
    "hamming_matrix",
    "build_db",
    "load_db",
    "query",
    "VideoFingerprint",
    "QueryResult",
]
