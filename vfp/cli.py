"""videofp 命令行入口：``build`` / ``query`` / ``eval`` 三个子命令。

用法::

    python -m vfp.cli build --sources "data/corpus/sources/*.mp4" --db db
    python -m vfp.cli query --db db --query data/corpus/queries/q001.mp4
    python -m vfp.cli eval  --db db --gt data/corpus/groundtruth.csv --report results/full.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from .eval import evaluate, load_gt, threshold_sweep, write_report
from .index import Config, build_db, load_db, query

__all__ = ["main", "build_parser"]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fps", type=float, default=2.0, help="抽帧率（默认 2.0）")
    p.add_argument("--threshold", type=int, default=10, help="帧级 Hamming 命中阈值")
    p.add_argument("--target-w", type=int, default=320, help="解码缩放宽度")
    p.add_argument("--no-crop-black", action="store_true", help="关闭黑边裁剪归一化（消融用）")
    p.add_argument("--no-mirror", action="store_true", help="关闭镜像双查询（消融用）")
    p.add_argument("--no-audio", action="store_true", help="关闭音频通道（消融用）")
    p.add_argument("--no-speed-search", action="store_true", help="关闭倍速搜索（消融用）")


def _cfg_from_args(a: argparse.Namespace) -> Config:
    cfg = Config(
        fps=a.fps,
        match_threshold=a.threshold,
        target_w=a.target_w,
        crop_black=not a.no_crop_black,
        mirror_query=not a.no_mirror,
        audio_enabled=not a.no_audio,
    )
    if getattr(a, "no_speed_search", False):
        cfg.speed_search = (1.0,)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vfp", description="视频盗版/抄袭鉴定系统（pHash 指纹）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="对参考片建库")
    b.add_argument("--sources", required=True, help='源片 glob，如 "data/corpus/sources/*.mp4"')
    b.add_argument("--db", default="db", help="指纹库输出目录")
    _add_common(b)

    q = sub.add_parser("query", help="查询一条待检视频")
    q.add_argument("--db", default="db")
    q.add_argument("--query", required=True, help="待检视频路径")
    q.add_argument("--json-out", default=None, help="把结果写成 JSON")
    _add_common(q)

    e = sub.add_parser("eval", help="批量评测并产出报告")
    e.add_argument("--db", default="db")
    e.add_argument("--gt", required=True, help="groundtruth.csv 路径")
    e.add_argument("--report", default="results/full.json")
    e.add_argument("--qdir", default=None, help="待检片目录（默认为 gt 同级的 queries/）")
    e.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部，调试用）")
    e.add_argument("--score-threshold", type=float, default=0.20, help="分数判定阈值")
    e.add_argument("--sweep", action="store_true", help="附带离线阈值扫描")
    _add_common(e)
    return ap


# --------------------------------------------------------------------------


def cmd_build(a: argparse.Namespace) -> int:
    files = sorted(glob.glob(a.sources))
    if not files:
        print(f"!! 没有匹配到源片: {a.sources}")
        return 1
    cfg = _cfg_from_args(a)
    print(f"建库: {len(files)} 条参考片 -> {a.db}")
    t0 = time.time()
    meta = build_db(files, a.db, cfg)
    print(f"\n完成: {meta['n_refs']} 条，用时 {time.time()-t0:.1f}s -> {a.db}/meta.json")
    return 0


def cmd_query(a: argparse.Namespace) -> int:
    db = load_db(a.db)
    cfg = _cfg_from_args(a)
    t0 = time.time()
    res = query(a.query, db, cfg)
    el = time.time() - t0

    print("=" * 66)
    print(f"查询: {res.qid}   ({el:.2f}s)")
    print("=" * 66)
    print(f"  判定      : {'命中' if res.matched else '未命中'}")
    print(f"  最高分    : {res.top_score:.4f}   (方法: {res.method})")
    print(f"  视觉/音频 : {res.video_score:.4f} / {res.audio_score:.4f}")
    print(f"  命中源片  : {res.source}")
    print(f"  命中帧数  : {res.n_hits} / {res.n_frames}   覆盖率 {res.coverage:.3f}")
    print(f"  平均距离  : {res.mean_dist:.2f}   倍速 {res.speed:.4f}   镜像 {res.mirrored}")
    print(f"  解码      : {res.decode_note}  (可用率 {res.usable_ratio:.3f})")
    if res.error:
        print(f"  [error]   : {res.error}")
    if res.segments:
        print("  时间码证据:")
        for s in res.segments:
            print(f"    query {s['q_start']:6.2f}-{s['q_end']:6.2f}s  "
                  f"<-> ref {s['r_start']:6.2f}-{s['r_end']:6.2f}s  "
                  f"帧 {s['n_frames']:3d}  dist {s['mean_dist']:.2f}  conf {s['confidence']:.3f}")
    else:
        print("  时间码证据: （无）")

    if a.json_out:
        write_report(a.json_out, res.as_dict())
        print(f"\n  JSON -> {a.json_out}")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    db = load_db(a.db)
    cfg = _cfg_from_args(a)
    gt = load_gt(a.gt)
    qdir = Path(a.qdir) if a.qdir else Path(a.gt).resolve().parent / "queries"
    rows = gt[: a.limit] if a.limit else gt

    print(f"评测: {len(rows)} 条 query，qdir={qdir}")
    print(f"配置: fps={cfg.fps} threshold={cfg.match_threshold} crop={cfg.crop_black} "
          f"mirror={cfg.mirror_query} audio={cfg.audio_enabled}")

    results: list[dict] = []
    t0 = time.time()
    for i, g in enumerate(rows, 1):
        p = qdir / g.file
        if not p.is_file():
            print(f"  [{i}/{len(rows)}] {g.qid}: 文件不存在 {p}")
            results.append({"qid": g.qid, "path": str(p), "top_score": 0.0,
                            "error": "文件不存在", "n_frames": 0})
            continue
        r = query(str(p), db, cfg, qid=g.qid)
        results.append(r.as_dict())
        flag = "命中" if r.matched else "未中"
        print(f"  [{i}/{len(rows)}] {g.qid} {g.transform[:18]:18s} -> {flag} "
              f"score={r.top_score:.4f} src={r.source} ({r.method})")
    elapsed = time.time() - t0

    metrics = evaluate(results, rows, threshold=a.score_threshold)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 2),
        "config": cfg.as_dict(),
        "metrics": metrics,
        "per_query": results,
    }
    if a.sweep:
        payload["threshold_sweep"] = threshold_sweep(results, rows)
    write_report(a.report, payload)

    print()
    print("=" * 66)
    print("指标")
    print("=" * 66)
    for k, v in metrics.items():
        print(f"  {k:32s} {v}")
    print(f"\n  用时 {elapsed:.1f}s ({elapsed/max(1,len(rows)):.2f}s/条) -> {a.report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            return cmd_build(a)
        if a.cmd == "query":
            return cmd_query(a)
        if a.cmd == "eval":
            return cmd_eval(a)
    except FileNotFoundError as e:
        print(f"!! {e}")
        return 1
    except KeyboardInterrupt:
        print("\n中断")
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
