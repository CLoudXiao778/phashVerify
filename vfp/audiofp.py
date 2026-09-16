"""音频指纹：星座图地标指纹 + 多尺度变速回放。

为什么必须做多尺度（Agent.md §9）：变速 1.05x 会让严格命中率从 0.998 掉到
0.608，1.25x 掉到 0.432。所以匹配时对参考音轨做 0.9/0.95/1.0/1.05/1.1/1.25
的变速重采样再比对，取最高分。

实现为"简化 Shazam"：谱图 -> 峰值 -> (f1, f2, dt) 组合哈希 -> 时间偏移直方图
投票。判定依据是**哈希在时间偏移上汇聚成尖峰**，而不是单条哈希命中率。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decode import ffmpeg_path, _NO_WINDOW  # noqa: F401  (_NO_WINDOW 供子模块参考)
import subprocess

__all__ = ["AudioFingerprint", "audio_fingerprint", "match_audio", "AudioMatch"]


# --------------------------------------------------------------------------
# 音频解码与谱图
# --------------------------------------------------------------------------

SR = 8000          # 采样率：够了，且省 CPU
N_FFT = 512
HOP = 128
PEAK_NEIGH = 12    # 峰值邻域半径（频率方向）
PEAK_FREQ_MIN = 2  # 忽略的 DC 低频 bin 数
FAN_OUT = 5        # 每个锚点配对的点数（Shazam 原论文建议 ~5）
DT_MIN = 1
DT_MAX = 64
FREQ_BITS = 9      # 频率量化位数 -> 匹配容错 ±1 格


def decode_audio_mono(path: str, sr: int = SR, timeout: int = 600) -> np.ndarray | None:
    """用 ffmpeg 解码为单声道 float32 PCM。无音轨/失败时返回 None（不抛异常）。"""
    cmd = [
        ffmpeg_path(),
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(path),
        "-vn", "-sn", "-dn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "-",
    ]
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=_NO_WINDOW, timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        return None
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).astype(np.float64)


def _spectrogram(sig: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """STFT 幅度谱 ``(n_bins, n_frames)``，用 Hann 窗。"""
    if len(sig) < n_fft:
        return np.zeros((n_fft // 2 + 1, 0))
    win = np.hanning(n_fft)
    n_frames = 1 + (len(sig) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = sig[idx] * win[None, :]
    spec = np.fft.rfft(frames, axis=1)  # (n_frames, n_bins)
    return np.abs(spec).T  # (n_bins, n_frames)


def _find_peaks(spec: np.ndarray) -> list[tuple[int, int]]:
    """局部最大值峰值检测，返回 ``[(freq_bin, time_frame), ...]``。"""
    if spec.size == 0:
        return []
    n_bins, n_frames = spec.shape
    log_spec = np.log1p(spec)
    peaks: list[tuple[int, int]] = []
    for t in range(n_frames):
        col = log_spec[:, t]
        for f in range(PEAK_FREQ_MIN, n_bins - 1):
            v = col[f]
            if v <= col[f - 1] or v < col[f + 1]:
                continue
            f0, f1 = max(0, f - PEAK_NEIGH), min(n_bins, f + PEAK_NEIGH + 1)
            t0, t1 = max(0, t - 3), min(n_frames, t + 4)
            if v >= log_spec[f0:f1, t0:t1].max():
                peaks.append((f, t))
    return peaks


@dataclass
class AudioFingerprint:
    """地标指纹。``hashes`` 为 ``(N,3)`` int32: (f1, f2, dt)。"""

    hashes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))
    times: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))
    n_peaks: int = 0
    duration: float = 0.0

    @property
    def n_hashes(self) -> int:
        return int(self.hashes.shape[0])

    def as_dict(self) -> dict:
        return {
            "n_hashes": self.n_hashes,
            "n_peaks": self.n_peaks,
            "duration": round(self.duration, 3),
        }


def audio_fingerprint(path: str, sr: int = SR) -> AudioFingerprint:
    """从音频文件/视频文件计算地标指纹。无音轨时返回空指纹（合法结果）。"""
    sig = decode_audio_mono(path, sr=sr)
    if sig is None or len(sig) < N_FFT:
        return AudioFingerprint()
    spec = _spectrogram(sig)
    peaks = _find_peaks(spec)
    if not peaks:
        return AudioFingerprint(n_peaks=0, duration=len(sig) / sr)

    # 按时间排序，构造 (f1, f2, dt) 组合哈希
    peaks.sort(key=lambda p: (p[1], p[0]))
    hashes: list[tuple[int, int, int]] = []
    times: list[int] = []
    n = len(peaks)
    for i in range(n):
        f1, t1 = peaks[i]
        target = t1 + DT_MAX
        paired = 0
        for j in range(i + 1, n):
            f2, t2 = peaks[j]
            if t2 > target:
                break
            dt = t2 - t1
            if dt < DT_MIN:
                continue
            hashes.append((f1, f2, dt))
            times.append(t1)
            paired += 1
            if paired >= FAN_OUT:
                break
    return AudioFingerprint(
        hashes=np.array(hashes, dtype=np.int32).reshape(-1, 3),
        times=np.array(times, dtype=np.int32),
        n_peaks=n,
        duration=len(sig) / sr,
    )


# --------------------------------------------------------------------------
# 匹配
# --------------------------------------------------------------------------


@dataclass
class AudioMatch:
    score: float = 0.0            # 汇聚峰值的归一化强度 (0-1)
    offset_frames: int = 0
    peak_votes: int = 0
    n_query_hashes: int = 0
    best_speed: float = 1.0

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "offset_frames": self.offset_frames,
            "peak_votes": self.peak_votes,
            "n_query_hashes": self.n_query_hashes,
            "best_speed": self.best_speed,
        }


def _hash_keys(h: np.ndarray, freq_bits: int = FREQ_BITS, dt_max: int = DT_MAX) -> np.ndarray:
    """把 (f1,f2,dt) 压成单个 int64 key，并额外给出频率 +1 的容错 key。"""
    f1, f2, dt = h[:, 0], h[:, 1], h[:, 2]
    mask = (1 << freq_bits) - 1
    return (f1.astype(np.int64) << (2 * freq_bits)) | ((f2.astype(np.int64) & mask) << freq_bits) | (dt.astype(np.int64) & mask)


def _vote(q: AudioFingerprint, r: AudioFingerprint, freq_tol: int) -> tuple[int, int]:
    """把 query 哈希在 ref 里查表，对时间偏移投票。

    返回 ``(peak_votes, best_offset)``。``freq_tol`` 控制频率量化容错格数。
    """
    if q.n_hashes == 0 or r.n_hashes == 0:
        return 0, 0

    # ref: key -> list of anchor times
    table: dict[int, list[int]] = {}
    keys_r = _hash_keys(r.hashes)
    for k, t in zip(keys_r.tolist(), r.times.tolist()):
        table.setdefault(k, []).append(t)

    keys_q = _hash_keys(q.hashes)
    offsets: list[int] = []
    mask = (1 << FREQ_BITS) - 1
    for k, tq in zip(keys_q.tolist(), q.times.tolist()):
        for tol in range(-freq_tol, freq_tol + 1):
            if tol == 0:
                kk = k
            else:
                # 只对 f2 的量化格做容错（f2 在 key 的低位区）
                f2 = (k >> FREQ_BITS) & mask
                f2n = f2 + tol
                if f2n < 0 or f2n > mask:
                    continue
                kk = (k & ~(mask << FREQ_BITS)) | (f2n << FREQ_BITS)
            for tr in table.get(kk, ()):  # type: ignore[arg-type]
                offsets.append(tr - tq)
    if not offsets:
        return 0, 0

    off = np.asarray(offsets, dtype=np.int64)
    # 时间偏移直方图投票，找峰值（±1 帧合并）
    vals, counts = np.unique(off, return_counts=True)
    best_i = int(np.argmax(counts))
    best_off = int(vals[best_i])
    peak = int(counts[best_i])
    # 合并相邻偏移的票数
    peak += int(counts[vals == best_off - 1].sum()) + int(counts[vals == best_off + 1].sum())
    return peak, best_off


def _resample_speed(sig: np.ndarray, speed: float, sr: int) -> np.ndarray:
    """线性插值重采样，模拟参考音轨被变速播放。"""
    if abs(speed - 1.0) < 1e-9:
        return sig
    n_out = int(len(sig) / speed)
    if n_out <= 0:
        return sig
    x = np.arange(n_out) * speed
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(sig) - 1)
    frac = x - i0
    i0 = np.clip(i0, 0, len(sig) - 1)
    return sig[i0] * (1.0 - frac) + sig[i1] * frac


def match_audio(
    query_path: str,
    ref_path: str,
    speeds: tuple[float, ...] = (1.0,),
    sr: int = SR,
    freq_tol: int = 1,
) -> AudioMatch:
    """比较两段音频。``speeds`` 给多个值即启用多尺度变速回放。"""
    q = audio_fingerprint(query_path, sr=sr)
    if q.n_hashes == 0:
        return AudioMatch(n_query_hashes=0)

    sig_r = decode_audio_mono(ref_path, sr=sr)
    if sig_r is None or len(sig_r) < N_FFT:
        return AudioMatch(n_query_hashes=q.n_hashes)

    best = AudioMatch(n_query_hashes=q.n_hashes)
    for sp in speeds:
        sig_rs = _resample_speed(sig_r, sp, sr)
        spec = _spectrogram(sig_rs)
        peaks = _find_peaks(spec)
        if not peaks:
            continue
        peaks.sort(key=lambda p: (p[1], p[0]))
        hashes, times = _peaks_to_hashes(peaks)
        r = AudioFingerprint(
            hashes=np.array(hashes, dtype=np.int32).reshape(-1, 3),
            times=np.array(times, dtype=np.int32),
            n_peaks=len(peaks),
            duration=len(sig_rs) / sr,
        )
        votes, off = _vote(q, r, freq_tol)
        if votes > best.peak_votes:
            best.peak_votes = votes
            best.offset_frames = off
            best.best_speed = float(sp)
    # 归一化：票数 / query 哈希数，再开方压缩（长音频哈希多，避免线性放大）
    if q.n_hashes > 0:
        best.score = float(min(1.0, np.sqrt(best.peak_votes / q.n_hashes)))
    return best


def _peaks_to_hashes(peaks: list[tuple[int, int]]) -> tuple[list[tuple[int, int, int]], list[int]]:
    """峰值列表 -> 组合哈希（与 ``audio_fingerprint`` 同一口径）。"""
    hashes: list[tuple[int, int, int]] = []
    times: list[int] = []
    n = len(peaks)
    for i in range(n):
        f1, t1 = peaks[i]
        target = t1 + DT_MAX
        paired = 0
        for j in range(i + 1, n):
            f2, t2 = peaks[j]
            if t2 > target:
                break
            dt = t2 - t1
            if dt < DT_MIN:
                continue
            hashes.append((f1, f2, dt))
            times.append(t1)
            paired += 1
            if paired >= FAN_OUT:
                break
    return hashes, times
