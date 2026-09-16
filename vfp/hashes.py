"""感知哈希实现：pHash64 / dct256 / dHash64 + 帧质量分。

设计约束（见 Agent.md C6，并已按实测修正）：
  - 依赖 ``numpy`` + ``Pillow``；DCT 走 ``scipy.fftpack``（**已实测存在的依赖**）
  - ``phash64`` 必须与 ``imagehash.phash`` 位级一致（可对照验证）
  - 全部为纯函数，同输入 -> 同输出（C7 可复现）

为什么用 scipy 的 DCT 而不是自研矩阵：
  自研的正交归一化 DCT 与 ``scipy.fftpack.dct`` 的默认（未归一化）形式相差一个
  逐行不同的缩放因子。中位数阈值虽然对整体缩放不敏感，但**系数之间的相对大小
  会变**，从而改变 20+ 个 bit 的判定结果（本机实测：4 张图共 112 bit 不一致）。
  为保证与社区实现位级一致（便于对照与审稿解释），这里直接用 scipy。
  ``scipy`` 本来就是 ``imagehash`` 的传递依赖，不新增实际负担。

坐标系约定：所有哈希以 ``numpy.uint64`` 数组承载，帧序列为 ``(N,)``，
一行位 = 一个 64 bit 无符号整数。
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fftpack import dct as _sp_dct

__all__ = [
    "phash64",
    "phash64_batch",
    "phash64_mirrored",
    "phash64_mirrored_batch",
    "dct256",
    "dhash64",
    "hamming64",
    "hamming_matrix",
    "laplacian_variance",
]

PHASH_DIM = 32       # hash_size(8) * highfreq_factor(4)
PHASH_LOW = 8        # 取 DCT 左上 8x8 低频块


def _to_gray_array(frame: Image.Image | np.ndarray, size: int) -> np.ndarray:
    """把输入统一转为 ``(size, size)`` 的 float64 灰度数组。"""
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim == 3:
            # RGB -> 灰度（与 Pillow 的 ITU-R 601-2 luma 权重一致）
            arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    else:
        img = frame.convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64)


def _dct2(x: np.ndarray) -> np.ndarray:
    """与 ``imagehash.phash`` 完全同口径的 2D DCT：先轴 0 后轴 1。"""
    return _sp_dct(_sp_dct(x, axis=0), axis=1)


def _phash_bits(frame: Image.Image | np.ndarray) -> np.ndarray:
    """返回 64 个 bool（row-major，MSB 在前），与 ``imagehash.phash`` 判定一致。

    关键细节：``med = numpy.median(dct[:8, :8])`` **包含 DC 系数**。
    """
    x = _to_gray_array(frame, PHASH_DIM)
    low = _dct2(x)[:PHASH_LOW, :PHASH_LOW]
    med = np.median(low)
    return (low > med).ravel()


def _bits_to_u64(bits: np.ndarray) -> np.uint64:
    """把 64 个 bool（MSB 在前）打包为 uint64。"""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)


def phash64(frame: Image.Image | np.ndarray) -> np.uint64:
    """计算 64 bit 感知哈希，与 ``imagehash.phash`` 位级一致。

    ``frame`` 可以是 Pillow ``Image`` 或 numpy 数组（HxW 灰度 或 HxWx3 RGB）。
    """
    return _bits_to_u64(_phash_bits(frame))


def phash64_batch(frames: np.ndarray) -> np.ndarray:
    """批量计算 phash64。

    参数
    ----
    frames : ``(N, H, W)`` uint8 灰度数组（H=W=32 时零拷贝；其他尺寸会缩放）

    返回
    ----
    ``(N,)`` 的 uint64 数组
    """
    if frames.ndim == 4:  # NHWC
        frames = frames[..., 0] * 0.299 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    n = len(frames)
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        out[i] = phash64(frames[i])
    return out


def dct256(frame: Image.Image | np.ndarray) -> np.ndarray:
    """256 bit 备用哈希：32x32 DCT 低 16x16 系数与中位数比较。

    返回 4 个 uint64（结构上等价于 256 bit）。相比 pHash 保留更多频域信息，
    对小水印/噪声更敏感，对几何攻击更脆弱 —— 具体优劣待 V7 实验数据。
    """
    x = _to_gray_array(frame, PHASH_DIM)
    full = _dct2(x)
    block = full[:16, :16].ravel()
    vals = block[1:]  # 丢弃 DC
    bits = vals > np.median(vals)
    # 255 bit 有效，补齐到 256
    bits = np.concatenate([bits, np.zeros(256 - len(bits), dtype=bool)])
    out = np.empty(4, dtype=np.uint64)
    for k in range(4):
        v = 0
        for b in bits[k * 64 : (k + 1) * 64]:
            v = (v << 1) | int(b)
        out[k] = np.uint64(v)
    return out


def dhash64(frame: Image.Image | np.ndarray) -> np.uint64:
    """64 bit 差分哈希：缩放到 9x8，比较水平相邻像素。"""
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim == 3:
            arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    else:
        img = frame.convert("L")
    small = np.asarray(img.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float64)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return _bits_to_u64(bits)


# --------------------------------------------------------------------------
# 距离度量
# --------------------------------------------------------------------------


def hamming64(a: np.uint64, b: np.uint64) -> int:
    """两个 uint64 哈希的 Hamming 距离。"""
    return int(np.uint64(a ^ b)).bit_count()


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(M,)`` x ``(N,)`` uint64 哈希的逐对 Hamming 距离矩阵 ``(M, N)``。

    用 numpy 的位运算 + popcount 查表实现，避免 Python 循环。
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    x = a[:, None] ^ b[None, :]
    # 4 个 16 bit 查表做 popcount
    lut = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
    x = x.reshape(x.shape[0], x.shape[1])
    out = np.zeros(x.shape, dtype=np.uint16)
    for shift in (0, 16, 32, 48):
        out += lut[((x >> np.uint64(shift)) & np.uint64(0xFFFF)).astype(np.uint16)]
    return out.astype(np.int32)


def mirror_hash(h: np.ndarray) -> np.ndarray:
    """**已废弃，请勿使用。** 仅保留以说明为什么不能这么做。

    早期设计假设"水平翻转后的 pHash == 原哈希按行位逆序"，这是**错的**。
    理由（已实测验证）：镜像后 DCT 系数在奇数列上会**翻转符号**，
    而 pHash 的每一位是"系数 > 中位数"的符号判定，因此系数位序与判定结果
    同时改变，不存在纯位置换的解。实测该函数置换后距离为 34 bit（等同随机）。

    正确做法见 ``phash64_mirrored``：在像素域翻转帧再算哈希。
    """
    raise NotImplementedError(
        "mirror_hash 在数学上不成立（见 docstring）。请使用 phash64_mirrored(pixel_frame)。"
    )


def phash64_mirrored(frame: Image.Image | np.ndarray) -> np.uint64:
    """对**已归一化的方阵帧**做水平翻转后再算 pHash。

    这是"镜像双查询"唯一正确的实现方式：先把帧缩放到同一尺寸（本管线为
    32x32 方阵），再左右翻转，最后算哈希。这样参考侧与查询侧的翻转发生在
    完全相同的处理阶段，两侧哈希可比。

    为什么不能在 64bit 哈希上做位运算见 ``mirror_hash``。
    """
    if isinstance(frame, np.ndarray):
        arr = np.asarray(frame)
        if arr.ndim == 3:
            arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        a = np.clip(arr, 0, 255).astype(np.uint8)
        size = a.shape[0]
        img = Image.fromarray(a, mode="L")
    else:
        img = frame.convert("L")
        size = img.size[0]
    # 先确保方阵与基线一致，再翻转
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return phash64(flipped)


def phash64_mirrored_batch(frames: np.ndarray) -> np.ndarray:
    """``(N,H,W)`` 归一化帧 -> 翻转后的 phash。"""
    n = len(frames)
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        out[i] = phash64_mirrored(frames[i])
    return out


def laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian 方差，作为清晰度/质量分（越大越清晰）。

    ``gray`` 为 HxW 灰度数组。用于在多帧候选里挑"最干净"的一帧。
    """
    g = gray.astype(np.float64)
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(lap.var()) if lap.size else 0.0
