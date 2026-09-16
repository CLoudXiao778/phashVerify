"""自动评测指标（防自欺协议，见 Agent.md §8）。

指标定义（与 §8.1 一致，不得自行改写口径）：

============== ==========================================================
recall         正样本中被判命中的比例
false_positive_rate  负样本中被判命中的比例
precision      TP / (TP + FP)
accuracy       全部样本判对的比例
source_top1_acc_on_positives  正样本中"最高分候选 == 真值源片"的比例
mean_iou_localization  仅在有 gt_segment 的行上算：预测片段与真值区间 IoU 均值
segment_f1     τ=0.5 下的片段级 P/R/F1（比 mean_IoU 更难被"糊一大段"骗过）
usable_ratio   解码可用帧数 / 总抽帧数（衡量损坏视频的降级程度）
============== ==========================================================

关键纪律（§8.2）：**阈值扫描必须离线用已存分数重算**，不要为每个阈值重跑匹配。
``threshold_sweep`` 就是干这个的。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .align import iou_f1, segment_iou

__all__ = ["load_gt", "evaluate", "threshold_sweep", "write_report"]

NEGATIVE_MARKERS = {"neg", "none", "negative", "无", "", "-"}


@dataclass
class GtRow:
    qid: str
    file: str
    source: str
    transform: str = ""
    verdict: str = ""
    gt_segment: tuple[float, float] | None = None
    corrupt: str = ""

    @property
    def is_negative(self) -> bool:
        """负样本判定：source 为空/neg/none，或 verdict 明确标为 negative。"""
        return self.source.strip().lower() in NEGATIVE_MARKERS or self.verdict.strip().lower().startswith("neg")


def _parse_segment(raw: str) -> tuple[float, float] | None:
    """解析 ``"12.5-20.0"`` 形式的时间区间。空/非法返回 None。"""
    if not raw or not raw.strip():
        return None
    txt = raw.strip().replace("~", "-").replace("–", "-")
    parts = txt.split("-")
    if len(parts) != 2:
        return None
    try:
        a, b = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (min(a, b), max(a, b))


def load_gt(path: str | Path) -> list[GtRow]:
    """读 groundtruth.csv。列：``qid,file,source,transform,verdict,gt_segment,corrupt``。"""
    rows: list[GtRow] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for rec in csv.DictReader(f):
            rows.append(
                GtRow(
                    qid=(rec.get("qid") or "").strip(),
                    file=(rec.get("file") or "").strip(),
                    source=(rec.get("source") or "").strip(),
                    transform=(rec.get("transform") or "").strip(),
                    verdict=(rec.get("verdict") or "").strip(),
                    gt_segment=_parse_segment(rec.get("gt_segment") or ""),
                    corrupt=(rec.get("corrupt") or "").strip(),
                )
            )
    return rows


def evaluate(results: list[dict], gt: list[GtRow], threshold: float = 0.20) -> dict:
    """按上面的口径计算指标。``results`` 为 ``QueryResult.as_dict()`` 的列表。"""
    by_qid = {r["qid"]: r for r in results}
    positives = [g for g in gt if not g.is_negative]
    negatives = [g for g in gt if g.is_negative]

    tp = fp = fn = tn = 0
    top1_ok = 0
    top1_den = 0
    ious: list[float] = []
    seg_p: list[float] = []
    seg_r: list[float] = []
    seg_f: list[float] = []
    usable: list[float] = []
    undecodable = 0
    recovered = 0
    unreachable: list[str] = []

    for g in gt:
        r = by_qid.get(g.qid)
        if r is None:
            unreachable.append(g.qid)
            continue
        sc = float(r.get("top_score") or 0.0)
        hit = sc >= threshold
        if g.is_negative:
            if hit:
                fp += 1
            else:
                tn += 1
        else:
            if hit:
                tp += 1
            else:
                fn += 1
            top1_den += 1
            src = (r.get("source") or "").strip()
            if src and src == g.source:
                top1_ok += 1
        if r.get("error") and not r.get("n_frames"):
            undecodable += 1
        if r.get("recovered"):
            recovered += 1
        usable.append(float(r.get("usable_ratio") or 0.0))

        # 定位指标：只对"有 gt_segment 且 gt 为正样本"的行计算
        if g.gt_segment is not None and not g.is_negative:
            pred = [(s["r_start"], s["r_end"]) for s in (r.get("segments") or [])]
            if pred:
                ious.append(max(segment_iou(p, g.gt_segment) for p in pred))
            else:
                ious.append(0.0)
            p_, r_, f_ = iou_f1(pred, [g.gt_segment], tau=0.5)
            seg_p.append(p_)
            seg_r.append(r_)
            seg_f.append(f_)

    n_pos, n_neg = len(positives), len(negatives)
    recall = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / (n_pos + n_neg) if (n_pos + n_neg) else 0.0

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "threshold": threshold,
        "n_queries": len(gt),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "precision": round(precision, 4),
        "accuracy": round(accuracy, 4),
        "source_top1_acc_on_positives": round(top1_ok / top1_den, 4) if top1_den else 0.0,
        "mean_iou_localization": round(_mean(ious), 4),
        "n_localization_rows": len(ious),
        "segment_precision_tau50": round(_mean(seg_p), 4),
        "segment_recall_tau50": round(_mean(seg_r), 4),
        "segment_f1_tau50": round(_mean(seg_f), 4),
        "mean_usable_ratio": round(_mean(usable), 4),
        "n_undecodable": undecodable,
        "n_recovered_by_fallback": recovered,
        "missing_qids": unreachable,
    }


def threshold_sweep(
    results: list[dict],
    gt: list[GtRow],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """离线阈值扫描：只用已存 ``top_score`` 重算 recall / FPR。

    §8.2 明确要求这么做 —— 为每个阈值重跑匹配既浪费又会引入不一致。
    """
    if thresholds is None:
        thresholds = [round(x / 100.0, 2) for x in range(0, 101, 5)]
    return [evaluate(results, gt, threshold=t) for t in thresholds]


def write_report(path: str | Path, payload: dict) -> None:
    """写 JSON 报告（UTF-8，不转义中文）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
