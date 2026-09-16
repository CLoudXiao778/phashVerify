"""序列对齐、片段定位、倍速估计、片段 IoU。

为什么必须做（Agent.md §9）：时序类攻击（剪一段、插广告）帧级存活率 100%，
但**朴素同轴逐帧比对会崩到 50%**（均值 Hamming 28.3）。必须做序列对齐。

为什么朴素对齐不够：变速会让 query 与 ref 的帧索引线性漂移。所以本模块在
"偏移搜索"之外还做 **倍速（scale）搜索**：speed != 1 时按线性映射选帧。
不处理变速，1.05x 的查询就会因为索引漂移而匹配不上。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hashes import hamming64, hamming_matrix

__all__ = [
    "MatchSegment",
    "AlignmentResult",
    "DIAG_HITS",
    "build_diag_hits",
    "segments_from_hits",
    "align_pair",
    "segment_iou",
    "iou_f1",
]


@dataclass
class MatchSegment:
    """一个匹配片段。时间码为秒，便于直接当证据引用。"""

    q_start: float
    q_end: float
    r_start: float
    r_end: float
    n_frames: int
    mean_dist: float
    confidence: float

    def as_dict(self) -> dict:
        return {
            "q_start": round(self.q_start, 3),
            "q_end": round(self.q_end, 3),
            "r_start": round(self.r_start, 3),
            "r_end": round(self.r_end, 3),
            "n_frames": self.n_frames,
            "mean_dist": round(self.mean_dist, 3),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class AlignmentResult:
    """一条 query 对一个参考片（或其镜像）的对齐结果。"""

    offset: int = 0          # ref_idx ≈ scale * q_idx + offset
    scale: float = 1.0
    offset_raw: int = 0      # 网格搜索得到的整数偏移（精修前）
    scale_raw: float = 1.0   # 网格搜索得到的候选倍速（精修前）
    speed_identifiable: bool = False  # 倍速是否唯一可辨识（否则不应采信精修）
    n_hits: int = 0
    n_query: int = 0
    mean_dist: float = 0.0
    coverage: float = 0.0
    segments: list[MatchSegment] = field(default_factory=list)
    mirrored: bool = False

    def as_dict(self) -> dict:
        return {
            "offset": self.offset,
            "scale": round(self.scale, 5),
            "offset_raw": self.offset_raw,
            "scale_raw": round(self.scale_raw, 5),
            "speed_identifiable": self.speed_identifiable,
            "n_hits": self.n_hits,
            "n_query": self.n_query,
            "mean_dist": round(self.mean_dist, 3),
            "coverage": round(self.coverage, 4),
            "mirrored": self.mirrored,
            "segments": [s.as_dict() for s in self.segments],
        }


DIAG_HITS = 12  # 偏移投票所需最少命中数（低于此不做对角线定位）


def build_diag_hits(
    q: np.ndarray,
    r: np.ndarray,
    threshold: int = 10,
    max_shift: int | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """构建对角线命中表::

        {shift: [(q_idx, r_idx), ...]}

    其中 ``shift = r_idx - q_idx``。这个函数是后面所有对齐/定位的基础。

    参数
    ----
    q, r : ``(N,)`` / ``(M,)`` uint64 哈希
    threshold : Hamming 命中阈值
    max_shift : 只保留 ``|shift| <= max_shift`` 的对角线（None 表示不限）
    """
    q = np.asarray(q, dtype=np.uint64)
    r = np.asarray(r, dtype=np.uint64)
    hits: dict[int, list[tuple[int, int]]] = {}
    if q.size == 0 or r.size == 0:
        return hits

    # 一次性算出全部距离矩阵，再按阈值过滤（比逐个 j 循环快）
    d = hamming_matrix(q, r)  # (N, M)
    qi, ri = np.nonzero(d <= threshold)
    if max_shift is not None:
        keep = np.abs(ri - qi) <= max_shift
        qi, ri = qi[keep], ri[keep]
    for a, b in zip(qi.tolist(), ri.tolist()):
        hits.setdefault(b - a, []).append((a, b))
    return hits


def _best_shift(hits: dict[int, list[tuple[int, int]]]) -> tuple[int, int]:
    """返回命中数最多的 ``(shift, n_hits)``。"""
    if not hits:
        return 0, 0
    shift = max(hits, key=lambda s: len(hits[s]))
    return shift, len(hits[shift])


def segments_from_hits(
    hits_list: list[tuple[int, int]],
    fps: float,
    q_len: int,
    r_len: int,
    max_gap: int = 3,
    min_frames: int = 2,
    conf_cap: float = 4.0,
) -> list[MatchSegment]:
    """把同一对角线上按顺序排列的命中点聚成连续片段。

    参数
    ----
    hits_list : ``[(q_idx, r_idx), ...]``，需按 ``q_idx`` 升序
    max_gap : 允许的帧级空档（抗丢帧/局部损坏）
    min_frames : 片段最少帧数
    conf_cap : 置信度归一化上限（``n_frames / conf_cap``，再乘密度因子）
    """
    segs: list[MatchSegment] = []
    if not hits_list:
        return segs

    hits_list = sorted(hits_list, key=lambda t: t[0])
    cur: list[tuple[int, int]] = [hits_list[0]]
    for h in hits_list[1:]:
        # 同时限制在 query 与 ref 轴上的间距，避免跨越"插广告"把两段接成一段
        if h[0] - cur[-1][0] <= max_gap and h[1] - cur[-1][1] <= max_gap:
            cur.append(h)
        else:
            segs.append(_make_segment(cur, fps, conf_cap))
            cur = [h]
    segs.append(_make_segment(cur, fps, conf_cap))

    return [s for s in segs if s.n_frames >= min_frames]


def _make_segment(hits: list[tuple[int, int]], fps: float, conf_cap: float) -> MatchSegment:
    qs = [h[0] for h in hits]
    rs = [h[1] for h in hits]
    span_q = qs[-1] - qs[0] + 1
    n = len(hits)
    density = n / span_q if span_q else 0.0
    conf = min(1.0, (n / conf_cap)) * density
    return MatchSegment(
        q_start=qs[0] / fps,
        q_end=(qs[-1] + 1) / fps,
        r_start=rs[0] / fps,
        r_end=(rs[-1] + 1) / fps,
        n_frames=n,
        mean_dist=0.0,  # 由调用方按真实距离回填
        confidence=conf,
    )


def _diag_candidates(
    q: np.ndarray, r: np.ndarray, scale: float, threshold: int, topk: int = 4
) -> tuple[list[tuple[int, int]], int, int]:
    """在给定倍速下投票选偏移，并返回该偏移上的命中对。

    返回 ``(hit_pairs, offset, n_hits)``。

    算法（这是本模块的核心，一个早期版本在这里写错过）：
      1. 对每个 query 帧 i，取它在参考序列上最接近的 ``topk`` 个候选帧 j
      2. 只有 ``hamming(q[i], r[j]) <= threshold`` 的候选才计入
      3. 对 ``offset = j - round(i * scale)`` 做直方图投票
      4. 票数最高的 offset 即最佳偏移；再在 ``±1`` 邻域收集命中，容忍取整误差

    **早期 bug 说明**：初版按"参考帧下标"分组统计命中，等于假设 offset 恒为 0，
    导致任何 scale 都得到同一个偏移、正确倍速选不出来（实测所有 scale 都返回
    offset=7/hits=1）。必须按 *offset* 而不是按 *参考帧下标* 分组。
    """
    n = len(q)
    m = len(r)
    if n == 0 or m == 0:
        return [], 0, 0

    idx = np.rint(np.arange(n) * scale).astype(np.int64)  # 预测的参考帧下标
    d = hamming_matrix(q, r)  # (n, m)

    k = min(topk, m)
    # argpartition 取每行最小的 k 个，避免全量排序
    part = np.argpartition(d, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = part.ravel()
    dists = d[rows, cols]
    keep = dists <= threshold
    rows, cols, dists = rows[keep], cols[keep], dists[keep]
    if rows.size == 0:
        return [], 0, 0

    # offset 直方图投票
    offs = cols - idx[rows]
    vals, counts = np.unique(offs, return_counts=True)
    offset = int(vals[int(np.argmax(counts))])

    # 在最佳 offset 的 ±1 邻域收集命中（每个 query 帧只取最近的一个）
    pred = idx + offset
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        j0 = int(pred[i])
        best_j, best_d = -1, 10**9
        for j in (j0 - 1, j0, j0 + 1):
            if 0 <= j < m:
                dd = int(d[i, j])
                if dd <= threshold and dd < best_d:
                    best_j, best_d = j, dd
        if best_j >= 0:
            pairs.append((i, best_j))
    return pairs, offset, len(pairs)


def _offset_for_scale(
    q: np.ndarray, r: np.ndarray, scale: float, threshold: int
) -> tuple[int, int]:
    """在给定倍速下找最佳偏移，返回 ``(offset, n_hits)``。"""
    _, offset, n_hits = _diag_candidates(q, r, scale, threshold)
    return offset, n_hits


def refine_speed(
    hit_pairs: list[tuple[int, int]],
    q_len: int,
    scale: float,
    offset: int,
    min_hits: int = 8,
    min_coverage: float = 0.5,
    neck: int = 2,
) -> tuple[float, float]:
    """用命中的 ``(q_idx, r_idx)`` 稳健估计倍速与偏移。

    为什么需要：候选倍速是离散网格（0.9/0.95/1.0/1.05/...），当画面帧间变化
    小时 1.0 与 1.05 都可能拿满命中，网格分辨率不足以区分。

    为什么**不用**普通最小二乘：真实映射是
    ``r_idx = round(scale * q_idx + offset)``，取整误差使 r_idx 呈阶梯状。
    对阶梯序列直接做 OLS 会系统性高估斜率（实测把真值 1.05 估成 1.089），
    这是典型的 errors-in-variables 偏差。

    也**不能**只用一次首尾差：区间太短时取整误差被放大（实测 20 点只能
    把 1.0526 估到 1.0）。

    这里用**多尺度分层平均**：对若干间隔 ``k``，算所有相距 ``k`` 的命中对的
    平均斜率 ``(r[i+k]-r[i])/(q[i+k]-q[i])``，再用命中数加权平均。
    间隔越大越不受取整噪声影响，间隔越小样本越多，加权后兼顾两者。
    """
    if len(hit_pairs) < min_hits:
        return float(scale), float(offset)
    if q_len > 0 and len(hit_pairs) / q_len < min_coverage:
        return float(scale), float(offset)

    pairs = sorted(hit_pairs, key=lambda t: t[0])
    qs = np.array([p[0] for p in pairs], dtype=np.float64)
    rs = np.array([p[1] for p in pairs], dtype=np.float64)
    n = len(pairs)
    if n < 4 or qs[-1] - qs[0] < 4:
        return float(scale), float(offset)

    ks = [k for k in (1, 2, 3, 5, 8, 13) if n - k >= 2]
    if not ks:
        return float(scale), float(offset)

    slopes, weights = [], []
    for k in ks:
        dq = qs[k:] - qs[:-k]
        dr = rs[k:] - rs[:-k]
        valid = dq > 0
        if not valid.any():
            continue
        slopes.append(float(np.mean(dr[valid] / dq[valid])))
        weights.append(float(valid.sum()))

    if not slopes:
        return float(scale), float(offset)
    w = np.array(weights, dtype=np.float64)
    a = float(np.sum(np.array(slopes) * w) / np.sum(w))
    if not (0.25 <= a <= 4.0) or not np.isfinite(a):
        return float(scale), float(offset)

    # 偏移取残差中位数（对离群点稳健），并做单帧微调
    resid = rs - a * qs
    b = float(np.median(resid))
    if n >= 4:
        cands = np.arange(-1.0, 1.51, 0.5)
        errs = [float(np.mean(np.abs(rs - np.round(a * qs + b + c)))) for c in cands]
        b += float(cands[int(np.argmin(errs))])
    return a, b


def align_pair(
    q: np.ndarray,
    r: np.ndarray,
    fps: float,
    threshold: int = 10,
    speed_candidates: tuple[float, ...] = (1.0,),
    max_gap: int = 3,
    min_frames: int = 2,
) -> AlignmentResult:
    """把一条 query 的哈希序列对齐到一个参考片的哈希序列。

    流程：对每个候选倍速投票选偏移 -> 取命中最多者 -> 在该对角线(±1)上
    收集命中 -> 聚成片段。同分时**偏好更接近 1.0 的倍速**，避免在对称的
    退化情形下无谓地选到 0.95 或 1.05（真实视频极少精确变速）。

    参数
    ----
    speed_candidates : 候选倍速。只搜 1.0 时退化为纯偏移对齐。
    """
    q = np.asarray(q, dtype=np.uint64)
    r = np.asarray(r, dtype=np.uint64)
    res = AlignmentResult(n_query=int(q.size))
    if q.size == 0 or r.size == 0:
        return res

    # 逐个候选倍速跑一遍，保留全部结果
    cands: list[tuple[int, float, int, list[tuple[int, int]]]] = []
    for sc in speed_candidates:
        pairs, off, cnt = _diag_candidates(q, r, sc, threshold)
        cands.append((cnt, float(sc), off, pairs))
    if not cands:
        return res

    best_cnt = max(c[0] for c in cands)
    top = [c for c in cands if c[0] == best_cnt]
    # 打平时偏好 |speed-1| 更小者
    top.sort(key=lambda c: abs(c[1] - 1.0))
    n_hits, scale, offset, hit_pairs = top[0]

    # 精修倍速前先做**可辨识性检查**。
    #
    # 原理：候选倍速是离散网格，若第一名与第二名拿到**相同命中数**，说明当前
    # 数据区分不开这两个倍速，斜率在信息论意义上不可辨识。此时任何"更精确"的
    # 精修都是过拟合 —— 实测 q004（12 帧 / 6 秒）上 1.0 与 1.05 并列第一，
    # 精修把倍速拉到 0.8457，结果把 6 秒的素材报成 5 秒区间，时间码算错。
    #
    # 因此：只有倍速**唯一可辨识**时才采纳精修结果，否则原样保留网格候选。
    identifiable = len(top) == 1
    res.scale_raw = float(scale)
    res.offset_raw = int(offset)
    if identifiable:
        scale_ref, off_ref = refine_speed(hit_pairs, int(q.size), scale, offset)
        res.scale = float(scale_ref)
        res.offset = int(round(off_ref))
    else:
        res.scale = float(scale)
        res.offset = int(offset)
    res.speed_identifiable = bool(identifiable)

    res.n_hits = int(n_hits)
    res.coverage = res.n_hits / q.size

    if hit_pairs:
        segs = segments_from_hits(hit_pairs, fps, q.size, r.size, max_gap, min_frames)
        for s in segs:
            i0 = int(round(s.q_start * fps))
            i1 = int(round(s.q_end * fps))
            ds = [int(dd) for (i, j) in hit_pairs if i0 <= i < i1
                  for dd in [hamming64(q[i], r[j])]]
            s.mean_dist = float(np.mean(ds)) if ds else 0.0
        res.segments = segs
        alld = [hamming64(q[i], r[j]) for (i, j) in hit_pairs]
        res.mean_dist = float(np.mean(alld)) if alld else 0.0
    return res


# --------------------------------------------------------------------------
# 评估用：片段 IoU
# --------------------------------------------------------------------------


def segment_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """两个时间区间的 IoU。"""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0:
        return 0.0
    return inter / union


def iou_f1(pred: list[tuple[float, float]], gt: list[tuple[float, float]], tau: float = 0.5) -> tuple[float, float, float]:
    """片段级 Precision / Recall / F1（IoU >= tau 视为命中）。

    定位评测不能只看总 IoU 均值：把两段预测成一大段也能骗到不低的均值。
    必须用 τ 阈值下的 P/R/F1，这是 VCDB/VCSL 一类工作的通行口径。
    """
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    matched_gt: set[int] = set()
    tp = 0
    for p in pred:
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            v = segment_iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= tau:
            matched_gt.add(best_j)
            tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1
