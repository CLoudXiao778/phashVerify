"""几何归一化：黑边检测裁剪 + 比例归一化。

为什么必须做（Agent.md §9）：加黑边（letterbox）会让 pHash 帧级命中率从
32/32 掉到 **1/32**；裁剪归一化后回到 32/32。所以黑边裁剪是主流程的必需步骤，
不是可选优化。

实现要点：本模块在**已解码的灰度帧数组**上工作，不重新调用 ffmpeg —— 这样
一份解码结果可以同时用于"归一化开/关"的消融对比，避免重复解码。
"""

from __future__ import annotations

import numpy as np

__all__ = ["find_black_border", "crop_frames", "normalize_frames"]


def find_black_border(
    frames: np.ndarray,
    thresh: int = 24,
    max_ratio: float = 0.30,
) -> tuple[int, int, int, int]:
    """检测恒定黑边，返回 ``(top, bottom, left, right)`` 四边各裁掉多少像素。

    判定口径：对**逐像素在时间轴上的最小值**取边缘均值，若某边界行的均值低于
    ``thresh`` 就认为该行是黑边。用逐像素最小值可以正确识别
    "黑边稳定、画面波动" 的 letterbox/pillarbox 情况。

    参数
    ----
    frames : ``(N, H, W)`` uint8
    thresh : 黑边亮度阈值（0-255）
    max_ratio : 单边最多裁掉的比例，防止把正常画面误裁

    注意：若视频本身整体很暗，本函数可能误裁。当前语料为亮画面，
    ``max_ratio`` 上限提供了兜底保护。
    """
    if frames.ndim != 3 or frames.shape[0] == 0:
        return (0, 0, 0, 0)

    n, h, w = frames.shape
    # 逐像素时间最小值 -> (H, W)
    pix_min = frames.min(axis=0).astype(np.float64)
    row_mean = pix_min.mean(axis=1)  # (H,)
    col_mean = pix_min.mean(axis=0)  # (W,)

    max_t = int(h * max_ratio)
    max_l = int(w * max_ratio)

    top = 0
    while top < max_t and row_mean[top] < thresh:
        top += 1
    bottom = 0
    while bottom < max_t and row_mean[h - 1 - bottom] < thresh:
        bottom += 1
    left = 0
    while left < max_l and col_mean[left] < thresh:
        left += 1
    right = 0
    while right < max_l and col_mean[w - 1 - right] < thresh:
        right += 1

    # 保护：裁完后至少保留 50% 边长
    if (h - top - bottom) < h * 0.5:
        top = bottom = 0
    if (w - left - right) < w * 0.5:
        left = right = 0
    return (top, bottom, left, right)


def crop_frames(frames: np.ndarray, border: tuple[int, int, int, int]) -> np.ndarray:
    """按 ``(top, bottom, left, right)`` 裁剪帧序列。"""
    top, bottom, left, right = border
    if top == bottom == left == right == 0:
        return frames
    h, w = frames.shape[1], frames.shape[2]
    y0, y1 = top, h - bottom
    x0, x1 = left, w - right
    if y1 <= y0 or x1 <= x0:
        return frames
    return frames[:, y0:y1, x0:x1]


def _resize_batch(frames: np.ndarray, size: int) -> np.ndarray:
    """把 ``(N,H,W)`` 批量缩放为 ``(N,size,size)``，返回 uint8。

    用 Pillow 的 LANCZOS 逐帧处理（与 ``hashes._to_gray_array`` 口径一致）。
    """
    from PIL import Image

    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    out = np.empty((frames.shape[0], size, size), dtype=np.uint8)
    for i in range(frames.shape[0]):
        im = Image.fromarray(frames[i], mode="L").resize((size, size), Image.Resampling.LANCZOS)
        out[i] = np.asarray(im, dtype=np.uint8)
    return out


def normalize_frames(
    frames: np.ndarray,
    crop_black: bool = True,
    square: bool = True,
    size: int = 32,
    thresh: int = 24,
    max_ratio: float = 0.30,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """黑边裁剪 + 方阵化归一化。

    返回 ``(归一化后的帧, 裁剪框)``。裁剪框会写进报告，便于审计
    "这条查询到底裁了多少"，避免静默改变输入。
    """
    border = find_black_border(frames, thresh=thresh, max_ratio=max_ratio) if crop_black else (0, 0, 0, 0)
    out = crop_frames(frames, border) if crop_black else frames
    if square:
        out = _resize_batch(out, size)
    return out, border
