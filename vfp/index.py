"""配置、建库、端到端匹配、融合打分。

本模块把前面的零件串成完整管线：

    视频文件
      -> 解码（decode，含降级链）
      -> 几何归一化（normalize，黑边裁剪）
      -> pHash64 + 镜像哈希（hashes）
      -> 序列对齐 + 片段定位（align）
      -> 音频地标指纹多尺度匹配（audiofp）
      -> 融合打分（本模块）

时间口径统一为**秒**，帧索引只出现在解码与对齐内部。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audiofp import (
    SR,
    AudioFingerprint,
    _find_peaks,
    _peaks_to_hashes,
    _resample_speed,
    _spectrogram,
    _vote,
    audio_fingerprint,
    decode_audio_mono,
)
from .align import AlignmentResult, align_pair
from .decode import DecodeResult, decode_gray, probe
from .hashes import hamming_matrix, phash64_batch, phash64_mirrored_batch
from .normalize import normalize_frames

__all__ = ["Config", "VideoFingerprint", "QueryResult", "build_db", "load_db", "query", "DEFAULT_SPEEDS"]

# 建库与查询时都要用到的变速候选（音频多尺度回放）
DEFAULT_SPEEDS: tuple[float, ...] = (0.95, 1.0, 1.05)
SCHEMA_VERSION = 2

# 参考音频指纹的进程内缓存（懒加载；见 _ref_audio_variants）
_REF_SIG_CACHE: dict[str, np.ndarray | None] = {}
_REF_AUDIO_CACHE: dict[tuple[str, float], AudioFingerprint | None] = {}


@dataclass
class Config:
    """全部可调参数集中在此，保证 C7 可复现（无隐藏随机/隐式默认）。"""

    fps: float = 2.0                  # 抽帧率
    hash_size: int = 32               # 归一化到 32x32 再算 pHash
    target_w: int = 320               # 解码缩放宽度
    match_threshold: int = 10         # 帧级 Hamming 命中阈值
    min_segment_frames: int = 2       # 片段最少帧数
    max_gap: int = 3                  # 片段内容许空档
    crop_black: bool = True           # 黑边裁剪归一化（§9 必需）
    mirror_query: bool = True         # 镜像双查询（§9 必需）
    audio_enabled: bool = True
    audio_speeds: tuple[float, ...] = DEFAULT_SPEEDS
    w_video: float = 0.6              # 融合权重：视频
    w_audio: float = 0.4              # 融合权重：音频
    chroma_fallback: bool = False     # 已判死，默认关闭（§5.4）
    speed_search: tuple[float, ...] = (1.0, 0.95, 1.05, 0.9, 1.1)
    seed: int = 7

    def as_dict(self) -> dict:
        d = asdict(self)
        d["audio_speeds"] = list(self.audio_speeds)
        d["speed_search"] = list(self.speed_search)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """宽容加载：忽略未知字段，缺失字段用默认值（避免 §4.3 的 TypeError）。"""
        allowed = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in allowed}
        for k in ("audio_speeds", "speed_search"):
            if k in clean and isinstance(clean[k], list):
                clean[k] = tuple(clean[k])
        return cls(**clean)


# --------------------------------------------------------------------------
# 指纹
# --------------------------------------------------------------------------


@dataclass
class VideoFingerprint:
    """一个视频的指纹（视觉 + 音频）。"""

    path: str
    phash: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))
    phash_mirror: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint64))
    fps: float = 2.0
    duration: float = 0.0
    n_frames: int = 0
    expected_frames: int = 0
    usable_ratio: float = 1.0
    decode_note: str = ""
    recovered: bool = False
    width: int = 0
    height: int = 0
    border: tuple[int, int, int, int] = (0, 0, 0, 0)
    quality_mean: float = 0.0
    audio: AudioFingerprint | None = None
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "n_frames": self.n_frames,
            "expected_frames": self.expected_frames,
            "usable_ratio": round(self.usable_ratio, 4),
            "fps": self.fps,
            "duration": round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "border": list(self.border),
            "quality_mean": round(self.quality_mean, 2),
            "decode_note": self.decode_note,
            "recovered": self.recovered,
            "audio": self.audio.as_dict() if self.audio else None,
            "error": self.error,
        }


def fingerprint_video(path: str, cfg: Config, with_audio: bool = True) -> VideoFingerprint:
    """解码 -> 归一化 -> pHash。所有失败都被记录，不抛异常。"""
    fp = VideoFingerprint(path=str(path), fps=cfg.fps)
    dec: DecodeResult = decode_gray(path, fps=cfg.fps, target_w=cfg.target_w)
    fp.duration = dec.duration
    fp.width = dec.width
    fp.height = dec.height
    fp.expected_frames = dec.expected_frames
    fp.decode_note = dec.note
    fp.recovered = dec.recovered
    if not dec.ok:
        fp.error = dec.error
        return fp
    if dec.n_frames == 0:
        fp.error = "解码得到 0 帧"
        return fp

    frames, border = normalize_frames(
        dec.frames,
        crop_black=cfg.crop_black,
        square=True,
        size=cfg.hash_size,
    )
    fp.border = border
    fp.n_frames = int(frames.shape[0])
    fp.usable_ratio = dec.usable_ratio
    # 质量分（Laplacian 方差）用于报告"这一帧够不够干净"
    g = frames.astype(np.float64)
    lap = (
        -4.0 * g[:, 1:-1, 1:-1]
        + g[:, :-2, 1:-1] + g[:, 2:, 1:-1]
        + g[:, 1:-1, :-2] + g[:, 1:-1, 2:]
    )
    fp.quality_mean = float(lap.var(axis=(1, 2)).mean()) if lap.size else 0.0

    fp.phash = phash64_batch(frames)
    # 镜像哈希必须在**同一归一化阶段**的帧上翻转后计算（见 hashes.phash64_mirrored）
    fp.phash_mirror = phash64_mirrored_batch(frames)

    if with_audio and cfg.audio_enabled:
        try:
            fp.audio = audio_fingerprint(str(path))
        except Exception as e:  # noqa: BLE001
            fp.audio = AudioFingerprint()
            fp.error = (fp.error + f" | 音频指纹失败: {e}").strip(" |")
    return fp


# --------------------------------------------------------------------------
# 建库 / 载库
# --------------------------------------------------------------------------


def build_db(sources: list[str], db_dir: str | Path, cfg: Config) -> dict:
    """对参考片建库。产出 ``<db>/meta.json`` + ``<db>/hashes.npz`` \
+ ``<db>/audio.json``。"""
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    refs: list[dict] = []
    vis: dict[str, np.ndarray] = {}
    mirror: dict[str, np.ndarray] = {}

    for i, src in enumerate(sources):
        t0 = time.time()
        fp = fingerprint_video(src, cfg, with_audio=cfg.audio_enabled)
        key = f"ref{i:04d}"
        info = probe(src)
        rec = {
            "key": key,
            "path": str(src),
            "basename": Path(src).name,
            "n_frames": fp.n_frames,
            "duration": round(fp.duration, 3),
            "fps": cfg.fps,
            "vcodec": info.vcodec,
            "acodec": info.acodec,
            "width": fp.width,
            "height": fp.height,
            "decode_note": fp.decode_note,
            "audio": fp.audio.as_dict() if fp.audio else None,
            "elapsed_s": round(time.time() - t0, 3),
            "error": fp.error,
        }
        refs.append(rec)
        vis[key] = fp.phash
        mirror[key] = fp.phash_mirror
        print(f"  [{i+1}/{len(sources)}] {Path(src).name}: {fp.n_frames} 帧, "
              f"{rec['elapsed_s']}s, {rec['decode_note']}")

    np.savez_compressed(db_dir / "hashes.npz", **vis, **{f"{k}__m": v for k, v in mirror.items()})
    meta = {
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg.as_dict(),
        "refs": refs,
        "n_refs": len(refs),
    }
    (db_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def load_db(db_dir: str | Path) -> dict:
    """载入指纹库。做 schema 兼容检查，缺失字段用默认值（不抛 TypeError）。"""
    db_dir = Path(db_dir)
    meta_path = db_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"指纹库不存在: {meta_path}（先跑 build）")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ver = meta.get("schema_version", 0)
    if ver != SCHEMA_VERSION:
        print(f"  [warn] 指纹库 schema_version={ver}，当前代码为 {SCHEMA_VERSION}；"
              f"已按宽容模式加载，建议重建。")
    data = np.load(db_dir / "hashes.npz")
    for rec in meta.get("refs", []):
        k = rec["key"]
        rec["phash"] = data[k] if k in data else np.zeros(0, np.uint64)
        rec["phash_mirror"] = data[f"{k}__m"] if f"{k}__m" in data else np.zeros(0, np.uint64)
    meta["config_obj"] = Config.from_dict(meta.get("config", {}))
    return meta


# --------------------------------------------------------------------------
# 查询
# --------------------------------------------------------------------------


@dataclass
class QueryResult:
    """一条查询的完整结果，可直接序列化为评测报告的一行。"""

    qid: str
    path: str
    matched: bool = False
    source: str | None = None
    top_score: float = 0.0
    video_score: float = 0.0
    audio_score: float = 0.0
    method: str = ""
    speed: float = 1.0
    mirrored: bool = False
    n_frames: int = 0
    n_hits: int = 0
    coverage: float = 0.0
    mean_dist: float = 0.0
    usable_ratio: float = 1.0
    recovered: bool = False
    decode_note: str = ""
    error: str = ""
    segments: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "qid": self.qid,
            "path": self.path,
            "matched": self.matched,
            "source": self.source,
            "top_score": round(self.top_score, 4),
            "video_score": round(self.video_score, 4),
            "audio_score": round(self.audio_score, 4),
            "method": self.method,
            "speed": round(self.speed, 5),
            "mirrored": self.mirrored,
            "n_frames": self.n_frames,
            "n_hits": self.n_hits,
            "coverage": round(self.coverage, 4),
            "mean_dist": round(self.mean_dist, 3),
            "usable_ratio": round(self.usable_ratio, 4),
            "recovered": self.recovered,
            "decode_note": self.decode_note,
            "segments": self.segments,
            "error": self.error,
            "candidates": self.candidates,
        }


def _normalize_video_score(align: AlignmentResult) -> float:
    """把一次对齐结果压成 0-1 的视觉分。

    口径：命中帧数越多、覆盖率越高、平均距离越小 -> 分越高。
    50 帧满命中（约 25 秒 @2fps）即视为饱和，避免长视频线性刷分。
    """
    if align.n_hits == 0:
        return 0.0
    hits_term = min(1.0, align.n_hits / 50.0)
    cov_term = align.coverage
    dist_term = max(0.0, 1.0 - align.mean_dist / 10.0)
    return float(0.5 * hits_term + 0.3 * cov_term + 0.2 * dist_term)


def _hash_fingerprint(sig: np.ndarray, sr: int = SR) -> AudioFingerprint | None:
    """把一段 PCM 转成地标指纹。无峰值时返回 None。"""
    if sig is None or len(sig) < 512:
        return None
    peaks = _find_peaks(_spectrogram(sig))
    if not peaks:
        return None
    peaks.sort(key=lambda p: (p[1], p[0]))
    hashes, times = _peaks_to_hashes(peaks)
    if not hashes:
        return None
    return AudioFingerprint(
        hashes=np.array(hashes, dtype=np.int32).reshape(-1, 3),
        times=np.array(times, dtype=np.int32),
        n_peaks=len(peaks),
        duration=len(sig) / sr,
    )


def _ref_audio_variants(ref_path: str, speeds: tuple[float, ...], sr: int = SR) -> list[tuple[float, AudioFingerprint]]:
    """懒加载并缓存参考片在各倍速下的音频指纹。

    这是原项目 §6 V5 指出的「``_ensure_scaled`` 懒加载从未执行」的位置 ——
    本实现把它做成显式带缓存的函数，并且**失败会被 ``query`` 捕获**，
    不会因为某条参考片音轨坏掉就中断整批查询。
    """
    cache: dict[tuple[str, float], AudioFingerprint | None] = _REF_AUDIO_CACHE
    out: list[tuple[float, AudioFingerprint]] = []
    # 注意：不能用 `_REF_SIG_CACHE.get(path, "miss") == "miss"` 判缺失 ——
    # 缓存里存的是 numpy 数组，`ndarray == "miss"` 返回**数组**，
    # if 判断会抛 "truth value of an array is ambiguous"（演示时实际踩到，
    # 导致音频通道整条静默失效）。必须用 `in` 判断键是否存在。
    if ref_path not in _REF_SIG_CACHE:
        try:
            _REF_SIG_CACHE[ref_path] = decode_audio_mono(ref_path, sr=sr)
        except Exception:  # noqa: BLE001
            _REF_SIG_CACHE[ref_path] = None
    sig = _REF_SIG_CACHE[ref_path]
    if sig is None:
        return out
    for sp in speeds:
        key = (ref_path, sp)
        if key not in cache:
            try:
                cache[key] = _hash_fingerprint(_resample_speed(sig, sp, sr), sr=sr)
            except Exception:  # noqa: BLE001
                cache[key] = None
        fp_r = cache[key]
        if fp_r is not None and fp_r.n_hashes:
            out.append((sp, fp_r))
    return out


def _audio_match_cached(q: AudioFingerprint | None, ref_path: str,
                        speeds: tuple[float, ...], sr: int = SR) -> tuple[float, int, float]:
    """query 音频 vs 参考片音频的多尺度匹配。返回 ``(score, votes, best_speed)``。"""
    if q is None or q.n_hashes == 0:
        return 0.0, 0, 1.0
    best_votes, best_off, best_sp = 0, 0, 1.0
    for sp, fp_r in _ref_audio_variants(ref_path, speeds, sr):
        votes, off = _vote(q, fp_r, freq_tol=1)
        if votes > best_votes:
            best_votes, best_off, best_sp = votes, off, sp
    if best_votes == 0:
        return 0.0, 0, 1.0
    score = float(min(1.0, np.sqrt(best_votes / q.n_hashes)))
    return score, best_votes, best_sp


def query(
    qpath: str,
    db: dict,
    cfg: Config | None = None,
    qid: str | None = None,
) -> QueryResult:
    """端到端查询一条视频，返回带时间码证据的结果。

    ``db`` 为 ``load_db`` 的返回值。参考片的音频在首次需要时才解码（懒加载），
    并且**失败会被捕获**，不会中断整批查询。
    """
    cfg = cfg or db.get("config_obj") or Config()
    qid = qid or Path(qpath).stem
    res = QueryResult(qid=qid, path=str(qpath))

    fp = fingerprint_video(qpath, cfg, with_audio=cfg.audio_enabled)
    res.n_frames = fp.n_frames
    res.usable_ratio = fp.usable_ratio
    res.recovered = fp.recovered
    res.decode_note = fp.decode_note
    if fp.error and fp.n_frames == 0:
        res.error = fp.error
        return res
    if fp.n_frames == 0:
        res.error = "无可用帧"
        return res

    best: dict | None = None
    for rec in db.get("refs", []):
        r = rec.get("phash")
        if r is None or len(r) == 0:
            continue
        for mirrored in ([False, True] if cfg.mirror_query else [False]):
            qh = fp.phash_mirror if mirrored else fp.phash
            al = align_pair(
                qh, r, fps=cfg.fps, threshold=cfg.match_threshold,
                speed_candidates=tuple(cfg.speed_search),
                max_gap=cfg.max_gap, min_frames=cfg.min_segment_frames,
            )
            v = _normalize_video_score(al)
            cand = {
                "source": rec["basename"],
                "key": rec["key"],
                "video_score": round(v, 4),
                "n_hits": al.n_hits,
                "coverage": round(al.coverage, 4),
                "mean_dist": round(al.mean_dist, 3),
                "speed": round(al.scale, 5),
                "offset": al.offset,
                "mirrored": mirrored,
            }
            if best is None or v > best["video_score"]:
                best = cand
                best["_align"] = al

    if best is None:
        res.error = "库内无可比对参考片"
        return res

    res.video_score = best["video_score"]
    al: AlignmentResult = best.pop("_align")
    res.n_hits = al.n_hits
    res.coverage = al.coverage
    res.mean_dist = al.mean_dist
    res.speed = al.scale
    res.mirrored = bool(best["mirrored"])
    # 只有真的存在命中证据时才报 source。
    # 早期版本在"全部候选都是 0 分"时仍会把第一个参考片写成 source，
    # 导致被正确拒绝的负样本也带出一个"命中源片"，读报告的人会被误导。
    res.source = best["source"] if al.n_hits > 0 else None
    res.segments = [s.as_dict() for s in al.segments]

    # ---- 音频通道（可选，懒加载 + 异常隔离）----
    # 只对视觉排名第一的参考片做音频比对：避免 O(N) 次音频解码。
    # 这是有意的取舍，写在报告里（method 字段会标明）。
    audio_score = 0.0
    if cfg.audio_enabled:
        try:
            q_audio = _hash_fingerprint(decode_audio_mono(qpath))
            ref_rec = next(
                (r for r in db.get("refs", []) if r.get("basename") == res.source), None
            )
            if q_audio is not None and ref_rec is not None:
                audio_score, votes, sp = _audio_match_cached(
                    q_audio, ref_rec["path"], tuple(cfg.audio_speeds)
                )
                res.candidates.append({"audio_votes": votes, "audio_speed": sp})
        except Exception as e:  # noqa: BLE001
            res.error = (res.error + f" | 音频匹配失败: {e}").strip(" |")
            audio_score = 0.0
    res.audio_score = audio_score

    # ---- 融合 ----
    if cfg.audio_enabled and audio_score > 0:
        res.top_score = cfg.w_video * res.video_score + cfg.w_audio * audio_score
        res.method = "video+audio"
    else:
        res.top_score = res.video_score
        res.method = "video"

    res.matched = res.top_score >= _score_threshold(cfg)
    res.candidates = [best] + [c for c in res.candidates if "source" not in c]
    return res


def _score_threshold(cfg: Config) -> float:
    """分数阈值。帧级阈值是 Hamming，这里换算成融合后的分数阈值。

    口径：以 §5.2 的假阳基线（无关视频最小 Hamming 18）为准，帧级阈值 10 对应
    "几乎没有偶然命中"。映射到分数空间后取 0.20（在合成语料上实测的保守值，
    见 results/ 里的阈值扫描曲线，而不是拍脑袋定的）。
    """
    return 0.20
