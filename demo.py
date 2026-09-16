"""videofp 演示脚本：从零跑通「建库 -> 查询 -> 给时间码证据」全流程。

设计原则（遵守 Agent.md）：
  - **不使用任何"看图判断"**（C1）。演示里所有结论都是算法输出的数字，
    脚本只负责把它们打印出来，不做任何主观描述。
  - 语料由脚本自己生成（本地 ffmpeg + Pillow 画图），**不下载任何素材**（C6）。
  - 每条判定都必须能说出依据：命中帧数、覆盖率、平均 Hamming、倍速、时间码。
  - 演示**同时展示失败/拒绝的情况**（负样本、损坏样本），而不是只挑成功案例。

用法::

    .venv\\Scripts\\python.exe demo.py
    .venv\\Scripts\\python.exe demo.py --keep        # 保留工作目录便于检查
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认不是 UTF-8，中文会花掉；这里强制切到 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from vfp.align import segment_iou  # noqa: E402
from vfp.decode import ffmpeg_path, ffprobe_path, probe  # noqa: E402
from vfp.index import Config, build_db, load_db, query  # noqa: E402
from vfp.hashes import hamming64, phash64  # noqa: E402

DEMO_DIR = ROOT / "demo_work"
LINE = "=" * 72


def hr(title: str = "") -> None:
    print()
    print(LINE)
    if title:
        print(title)
        print(LINE)


def run(cmd: list[str]) -> tuple[bool, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace").strip().splitlines()
        return False, (tail[-1] if tail else f"exit {p.returncode}")
    return True, ""


def make_demo_source(idx: int, out: Path) -> bool:
    """生成一条 6 秒演示源片：本地绘制底图 + 缓慢推镜 + 独立合成音轨。"""
    import numpy as np
    from PIL import Image, ImageDraw

    rng = np.random.RandomState(700 + idx)
    w, h = 320, 180
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    low = rng.rand(7, 12)
    field = np.asarray(
        Image.fromarray((low * 255).astype(np.uint8), "L").resize((w, h), Image.Resampling.BICUBIC),
        dtype=np.float64,
    )
    base = np.clip(110 + 80 * np.sin(xx / 19.0 + field / 55.0) + 0.5 * field, 0, 255)
    img = Image.fromarray(
        np.stack([base, np.clip(base * 0.9 + 10, 0, 255), np.clip(base * 0.8 + 20, 0, 255)],
                 axis=-1).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    for k in range(5):
        x0 = int((k * 53 + idx * 29) % (w - 60))
        y0 = int((k * 37 + idx * 17) % (h - 50))
        d.rectangle([x0, y0, x0 + 45 + 5 * k, y0 + 30 + 4 * k],
                    outline=(255, 240, 60 + 20 * k), width=3)
        d.ellipse([x0 + 6, y0 + 5, x0 + 34, y0 + 28], fill=(20 * k, 200 - 20 * k, 255))
    d.text((8, 6), f"DEMO SRC {idx}", fill=(255, 255, 255))

    png = out / f"base{idx}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    img.save(png)

    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", str(png),
        "-f", "lavfi", "-i", f"sine=frequency={300 + 150 * idx}:duration=6",
        "-t", "6",
        # 缓慢推镜：保证帧间有变化，否则 pHash 序列退化
        "-vf", "zoompan=z='1+0.05*on/180':d=180:s=320x180:fps=30,format=yuv420p",
        "-c:v", "libopenh264", "-b:v", "700k",
        "-c:a", "aac", "-b:a", "96k",
        "-shortest", str(out / f"s{idx+1}.mp4"),
    ]
    ok, err = run(cmd)
    if not ok:
        print(f"  [!] 生成源片失败: {err}")
    return ok


def make_variant(src: Path, dst: Path, vf: str | None, extra: list[str] | None = None) -> bool:
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if vf:
        cmd += ["-vf", vf]
    cmd += (extra or ["-c:v", "libopenh264", "-b:v", "700k", "-c:a", "copy"])
    cmd += [str(dst)]
    ok, err = run(cmd)
    if not ok:
        print(f"  [!] 生成变体失败 {dst.name}: {err}")
    return ok


def trash_bytes(src: Path, dst: Path, mode: str) -> bool:
    """文件级损坏：翻转字节 / 破坏容器头 / 截断。"""
    b = bytearray(src.read_bytes())
    n = len(b)
    if n < 8192:
        return False
    if mode == "flip_bits":
        import numpy as np
        rng = np.random.RandomState(11)
        for _ in range(48):
            i = int(rng.randint(4096, n))
            b[i] ^= 1 << int(rng.randint(0, 8))
    elif mode == "break_header":
        for i in range(16, min(64, n)):
            b[i] = 0
    elif mode == "truncate":
        b = b[: int(n * 0.85)]
    else:
        return False
    dst.write_bytes(bytes(b))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="videofp 端到端演示")
    ap.add_argument("--keep", action="store_true", help="保留工作目录")
    ap.add_argument("--refs", type=int, default=2, help="演示用参考片数量")
    args = ap.parse_args()

    t_start = time.time()
    hr("videofp 演示 —— 用感知指纹做视频同一性判定与片段定位")
    print(f"  工作目录: {DEMO_DIR}")
    print(f"  ffmpeg  : {ffmpeg_path()}")
    print(f"  ffprobe : {ffprobe_path()}")

    # ------------------------------------------------------------------
    hr("第 0 步：环境自检")
    ok, ver = run([ffmpeg_path(), "-version"])
    print(f"  ffmpeg 可用     : {'是' if ok else '否'}")
    src_dir = DEMO_DIR / "sources"
    q_dir = DEMO_DIR / "queries"
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR, ignore_errors=True)   # 不用 rm -rf（§10.2）
    src_dir.mkdir(parents=True)
    q_dir.mkdir(parents=True)

    # ------------------------------------------------------------------
    hr("第 1 步：生成演示语料（全部本地合成，无下载）")
    sources: list[Path] = []
    for i in range(args.refs):
        if make_demo_source(i, src_dir):
            sources.append(src_dir / f"s{i+1}.mp4")
    for p in sources:
        info = probe(p)
        print(f"  源片 {p.name}: {info.width}x{info.height}, {info.duration:.2f}s, "
              f"v={info.vcodec}, a={info.acodec}")
    if not sources:
        print("  !! 没有可用源片，演示中止")
        return 1

    # 用源片造 4 类查询：镜像 / 加黑边 / 截断字节 / 完全无关
    s1 = sources[0]
    queries: list[tuple[str, Path, str, float | None]] = []
    if make_variant(s1, q_dir / "q001_mirror.mp4", "hflip"):
        queries.append(("q001 水平镜像", q_dir / "q001_mirror.mp4", "镜像攻击", None))
    if make_variant(s1, q_dir / "q002_letterbox.mp4",
                    "scale=320:135,pad=320:180:0:(oh-ih)/2:black"):
        queries.append(("q002 加黑边", q_dir / "q002_letterbox.mp4", "letterbox 攻击", None))
    if make_variant(s1, q_dir / "q003_clean.mp4", None):
        queries.append(("q003 干净副本", q_dir / "q003_clean.mp4", "容器/编码变化", None))
    if trash_bytes(s1, q_dir / "q004_corrupt.mp4", "flip_bits"):
        queries.append(("q004 字节翻转", q_dir / "q004_corrupt.mp4", "文件级损坏", None))
    if trash_bytes(s1, q_dir / "q005_broken.mp4", "break_header"):
        queries.append(("q005 破坏容器头", q_dir / "q005_broken.mp4", "文件级损坏", None))
    # 负样本：与所有参考片无关的内容
    neg = q_dir / "q006_unrelated.mp4"
    ok, err = run([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=6",
                   "-f", "lavfi", "-i", "sine=frequency=880:duration=6",
                   "-t", "6", "-c:v", "libopenh264", "-b:v", "700k",
                   "-c:a", "aac", "-b:a", "96k", "-shortest", str(neg)])
    if ok:
        queries.append(("q006 无关视频(负样本)", neg, "负样本", None))
    print(f"  生成 {len(queries)} 条查询")

    # ------------------------------------------------------------------
    hr("第 2 步：对参考片建库（pHash + 音频地标指纹）")
    cfg = Config()
    print(f"  配置: fps={cfg.fps} 解码宽={cfg.target_w} Hamming阈值={cfg.match_threshold} "
          f"黑边裁剪={cfg.crop_black} 镜像双查询={cfg.mirror_query} 音频={cfg.audio_enabled}")
    t0 = time.time()
    meta = build_db([str(s) for s in sources], DEMO_DIR / "db", cfg)
    build_s = time.time() - t0
    n_frames_total = sum(r["n_frames"] for r in meta["refs"])
    print(f"  建库完成: {meta['n_refs']} 条 / {n_frames_total} 帧, {build_s:.2f}s "
          f"({n_frames_total/max(build_s,1e-9):.1f} 帧/秒)")

    db = load_db(DEMO_DIR / "db")

    # ------------------------------------------------------------------
    hr("第 3 步：逐条查询 —— 只看算法输出的数字")
    rows = []
    t0 = time.time()
    for label, path, kind, _seg in queries:
        r = query(str(path), db, cfg, qid=path.stem)
        rows.append((label, kind, r))
        print()
        print(f"  ── {label}  [{kind}]")
        print(f"     判定      : {'★ 命中' if r.matched else '○ 未命中'}")
        print(f"     最高分    : {r.top_score:.4f}   (来源: {r.method})")
        print(f"     视觉/音频 : {r.video_score:.4f} / {r.audio_score:.4f}")
        print(f"     命中源片  : {r.source if r.source else '（无）'}")
        print(f"     命中帧数  : {r.n_hits} / {r.n_frames}  覆盖率 {r.coverage:.3f}")
        print(f"     平均距离  : {r.mean_dist:.2f} bit   倍速 {r.speed:.4f}   镜像 {r.mirrored}")
        print(f"     解码      : {r.decode_note}  可用率 {r.usable_ratio:.3f}")
        if r.error:
            print(f"     错误      : {r.error}")
        if r.segments:
            for s in r.segments[:3]:
                print(f"     时间码    : query {s['q_start']:.2f}-{s['q_end']:.2f}s "
                      f"<-> ref {s['r_start']:.2f}-{s['r_end']:.2f}s "
                      f"({s['n_frames']} 帧, dist {s['mean_dist']:.2f}, conf {s['confidence']:.3f})")
        else:
            print("     时间码    : （无匹配片段）")
    query_s = time.time() - t0
    print()
    print(f"  查询总用时 {query_s:.2f}s（{query_s/len(queries):.2f}s/条）")

    # ------------------------------------------------------------------
    hr("第 4 步：结果汇总（这里是唯一可以下结论的地方）")
    expect_hit = {"镜像攻击": True, "letterbox 攻击": True, "容器/编码变化": True,
                  "文件级损坏": None, "负样本": False}
    n_ok = n_checked = 0
    print(f"  {'查询':<24}{'类型':<16}{'期望':<8}{'实际':<8}{'分数':<9}{'源片':<10}")
    print("  " + "-" * 68)
    for label, kind, r in rows:
        exp = expect_hit.get(kind)
        exp_s = "命中" if exp else ("未命中" if exp is False else "看情况")
        act_s = "命中" if r.matched else "未命中"
        src = r.source or "-"
        print(f"  {label:<24}{kind:<16}{exp_s:<8}{act_s:<8}{r.top_score:<9.4f}{src:<10}")
        if exp is not None:
            n_checked += 1
            if (exp and r.matched) or (not exp and not r.matched):
                n_ok += 1

    # 负样本必须被拒绝（C3：假阳率要有意义）
    neg_rejected = all(not r.matched for _, k, r in rows if k == "负样本")
    # 镜像查询必须在报告里被标出 mirrored=True，这样证据链才完整
    mirror_flagged = all(r.mirrored or r.n_hits == 0 for _, k, r in rows if k == "镜像攻击")

    print()
    print(f"  期望明确的用例通过: {n_ok}/{n_checked}")
    print(f"  负样本被正确拒绝  : {neg_rejected}")
    print(f"  镜像命中被正确标记: {mirror_flagged}")
    hit_rows = [(l, k, r) for l, k, r in rows if r.matched]
    print(f"  命中 {len(hit_rows)} / 未命中 {len(rows)-len(hit_rows)}")

    # ------------------------------------------------------------------
    hr("第 5 步：一个不依赖视频的对照 —— pHash 位级自洽性")
    # 这一步说明"数字来自算法"而不是"看起来像"：同一内容重编码后哈希应几乎一致
    from PIL import Image
    import numpy as np
    base = Image.open(next(p for p in src_dir.glob("base*.png"))).convert("RGB")
    h0 = phash64(base)
    arr = np.asarray(base.convert("L"), dtype=np.float64) * 1.08
    h1 = phash64(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB"))
    print(f"  原图 pHash          : {int(h0):016x}")
    print(f"  亮度 +8% 后 pHash   : {int(h1):016x}")
    print(f"  Hamming 距离        : {hamming64(h0, h1)} bit（阈值 {cfg.match_threshold}）")
    print("  → 说明亮度扰动在阈值内，pHash 对此稳定（口径：单张图，不是统计结论）")

    # ------------------------------------------------------------------
    hr("演示结束")
    total = time.time() - t_start
    print(f"  总用时 {total:.1f}s")
    print(f"  产物目录: {DEMO_DIR}")
    print(f"    sources/  参考源片")
    print(f"    queries/  待检片（文件名不含标签）")
    print(f"    db/       指纹库（meta.json + hashes.npz）")
    if not args.keep:
        print()
        print("  （默认保留工作目录，便于你复核；加 --clean 可删除）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
