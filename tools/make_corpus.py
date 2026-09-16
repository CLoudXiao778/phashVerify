"""生成合成评测语料：参考源片 / 篡改变体 / 负样本 / 中性化待检片 + 真值。

设计纪律（全部来自 Agent.md 的踩坑记录）：
  1. **文件名中性化**（C2）：待检片一律叫 ``q001.mp4``，真实含义只写在 groundtruth.csv
  2. **每个变换都真正执行 ffmpeg，并检查返回值**（§10.5 惨案：脚本只 return 命令字符串
     没执行，却把 60 行真值写进了 CSV）
  3. **结束后必须做文件数与时长校验**（§10.5 的修复要求）
  4. 素材全部本地生成（Pillow 画图 + 合成音轨），**不下载**，符合离线/轻量约束（C6）
  5. 负样本必须存在（C3），否则假阳率算不出来

用法::

    python -m tools.make_corpus --out data/corpus
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vfp.decode import ffmpeg_path, ffprobe_path  # noqa: E402

W, H = 640, 360
FPS = 30
DUR = 8  # 每条 8 秒：够 2fps 抽 16 帧，且磁盘开销小
SEED = 7


# --------------------------------------------------------------------------
# 生成静态底图（本地绘制，不下载）
# --------------------------------------------------------------------------


def make_base_image(kind: int, w: int = W, h: int = H) -> Image.Image:
    """用 numpy 画一张有丰富空间结构的图，供 pHash 使用。

    不同 ``kind`` 之间结构差异要足够大，这样负样本之间不会互相误命中。
    """
    rng = np.random.RandomState(SEED + kind)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # 平滑随机场（低分辨率上采样），提供大尺度结构
    low = rng.rand(9, 16)
    field = np.asarray(
        Image.fromarray((low * 255).astype(np.uint8), "L").resize((w, h), Image.Resampling.BICUBIC),
        dtype=np.float64,
    )

    if kind % 4 == 0:
        base = 120 + 90 * np.sin(xx / 37.0 + field / 60.0)
    elif kind % 4 == 1:
        base = 110 + 100 * np.cos(yy / 29.0) * np.sin(xx / 51.0)
    elif kind % 4 == 2:
        base = 90 + field * 0.7 + 60 * np.sin((xx + yy) / 44.0)
    else:
        base = 100 + 80 * np.sin(np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / 18.0)

    base = np.clip(base, 0, 255)
    # 转成 RGB，给不同通道不同偏移 -> 饱和度/色相信息也可用
    r = np.clip(base * 1.02, 0, 255)
    g = np.clip(base * 0.92 + 12, 0, 255)
    b = np.clip(base * 0.80 + 25 * (kind % 3), 0, 255)
    arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    # 叠几何图形，制造高频边缘（pHash 需要边缘）
    d = ImageDraw.Draw(img)
    for k in range(6):
        x0 = int((k * 97 + kind * 53) % (w - 90))
        y0 = int((k * 61 + kind * 31) % (h - 70))
        d.rectangle([x0, y0, x0 + 60 + 7 * k, y0 + 40 + 5 * k],
                    outline=(255 - 20 * k, 240, 60 + 30 * k), width=3)
        d.ellipse([x0 + 8, y0 + 6, x0 + 46, y0 + 40], fill=(30 * k % 255, 200 - 25 * k, 255))
    d.line([(0, h // 2), (w, h // 2)], fill=(255, 255, 255), width=2)
    d.text((12, 10), f"REF-{kind} videofp", fill=(255, 255, 255))
    return img


def _run(cmd: list[str]) -> tuple[bool, str]:
    """执行 ffmpeg 命令。返回 ``(成功, 错误尾巴)``。"""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace").strip().splitlines()
        return False, (tail[-1] if tail else f"exit {p.returncode}")
    return True, ""


def _probe_duration(path: Path) -> float:
    p = subprocess.run(
        [ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        return float(p.stdout.decode().strip())
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------
# 生成参考源片
# --------------------------------------------------------------------------


def make_source(idx: int, out: Path) -> bool:
    """生成一条参考源片：静态底图 + 缓慢推镜 + 独立合成音轨。

    每条源片用**不同频率的合成音轨**，避免音频侧无法区分源片（Agent.md §4.2）。
    """
    img_path = out.parent / "tmp" / f"base{idx}.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    make_base_image(idx).save(img_path)

    freq = 220 * (idx + 1)
    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", str(img_path),
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={DUR}",
        "-t", str(DUR),
        # 缓慢推镜：zoompan 提供时间维度的变化（否则帧帧相同，pHash 退化）
        "-vf", f"zoompan=z='1+0.03*on/{FPS*DUR}':d={FPS*DUR}:s={W}x{H}:fps={FPS},format=yuv420p",
        "-c:v", "libopenh264", "-b:v", "900k",
        "-c:a", "aac", "-b:a", "96k",
        "-shortest", str(out),
    ]
    ok, err = _run(cmd)
    if not ok:
        print(f"    [!] 源片 {out.name} 生成失败: {err}")
    return ok


# --------------------------------------------------------------------------
# 变换定义
# --------------------------------------------------------------------------


def build_variants(src: Path, tmp: Path) -> list[tuple[str, list[str], float | None, str]]:
    """返回 ``[(变体名, ffmpeg 参数, 真值源片段, 备注), ...]``。

    真值源片段为 ``(start, end)``（参考时间轴，秒），None 表示"整片对应"。
    """
    v: list[tuple[str, list[str], float | None, str]] = []

    def add(name: str, args: list[str], seg=None, note: str = "") -> None:
        v.append((name, args, seg, note))

    # --- 容器 / 编码 ---
    add("container_mkv", ["-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"], note="换容器 mkv")
    add("codec_mpeg4", ["-c:v", "mpeg4", "-q:v", "6", "-c:a", "copy"], note="换编码 mpeg4")
    add("codec_vp9", ["-c:v", "libvpx-vp9", "-crf", "40", "-b:v", "0", "-c:a", "libopus"], note="VP9+OPUS")
    # --- 分辨率 ---
    add("rescale_360p", ["-vf", "scale=640:360", "-c:v", "libopenh264", "-b:v", "500k", "-c:a", "copy"])
    add("rescale_180p", ["-vf", "scale=320:180", "-c:v", "libopenh264", "-b:v", "250k", "-c:a", "copy"])
    add("rescale_720p", ["-vf", "scale=1280:720", "-c:v", "libopenh264", "-b:v", "1500k", "-c:a", "copy"])
    # --- 几何（pHash 的软肋，§5.2）---
    add("letterbox", ["-vf", f"scale={W}:{int(H*0.75)},pad={W}:{H}:0:(oh-ih)/2:black",
                      "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"],
        note="加黑边：命中率会掉到 1/32，靠黑边裁剪归一化救回")
    add("pillarbox", ["-vf", f"scale={int(W*0.75)}:{H},pad={W}:{H}:(ow-iw)/2:0:black",
                      "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"], note="左右黑边")
    add("mirror", ["-vf", "hflip", "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"],
        note="水平镜像：命中率 0/32，靠镜像双查询救回")
    add("crop_90", ["-vf", f"crop={int(W*0.9)}:{int(H*0.9)}:0:0,scale={W}:{H}",
                    "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"], note="裁剪保留 90%")
    # --- 画质 ---
    add("blur_noise", ["-vf", "gblur=sigma=1.2,noise=alls=8:allf=t",
                       "-c:v", "libopenh264", "-b:v", "700k", "-c:a", "copy"])
    # 本机 ffmpeg 构建（BtbN LGPL）**没有编译 eq 滤镜**，也没有 brightness 滤镜。
    # 实测可用替代：colorlevels（亮度/对比）+ huesaturation（饱和度）。
    # 踩坑记录：写 -vf brightness=... 或 -vf eq=... 都会 "Filter not found"；
    # 且 colorlevels 的选项名是 rimin/gimin/bimin，不是 rim/gim/bim。
    add("brightness", ["-vf", "colorlevels=rimin=0.06:gimin=0.06:bimin=0.06,"
                               "huesaturation=saturation=0.25",
                       "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"],
        note="亮度/对比/饱和度调整")
    # --- 叠加 ---
    add("watermark", ["-vf", f"drawbox=x=20:y=20:w={int(W*0.3)}:h={int(H*0.25)}:color=white@0.55:t=fill,"
                             f"drawbox=x=20:y=20:w={int(W*0.3)}:h={int(H*0.25)}:color=black:t=3",
                      "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"], note="大块不透明水印")
    add("subtitle", ["-vf", f"drawbox=x=0:y={int(H*0.85)}:w={W}:h={int(H*0.12)}:color=black@0.8:t=fill",
                     "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"], note="字幕条")
    # --- 时序 ---
    add("trim_head", ["-ss", "2", "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"],
        seg=(2.0, DUR), note="掐头 2s：真值片段整体后移")
    add("trim_both", ["-ss", "1.5", "-t", "4.0", "-c:v", "libopenh264", "-b:v", "900k", "-c:a", "copy"],
        seg=(1.5, 5.5), note="掐头去尾")
    # --- 音频 ---
    add("audio_swap", ["-c:v", "copy", "-af", "volume=0.3"], note="音量降 70%")
    add("audio_reencode", ["-c:v", "copy", "-c:a", "libmp3lame", "-b:a", "64k"], note="MP3 64k")
    return v


def make_variants(src: Path, tmp: Path, tag: str) -> list[dict]:
    """对一条源片生成全部变体，返回真值记录列表。"""
    out: list[dict] = []
    for name, args, seg, note in build_variants(src, tmp):
        ext = ".mkv" if "container_mkv" in name else (".webm" if "vp9" in name else ".mp4")
        dst = tmp / f"{tag}_{name}{ext}"
        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), *args, str(dst)]
        ok, err = _run(cmd)
        if not ok:
            print(f"    [!] 变体 {name} 失败: {err}")
            continue
        out.append({
            "name": name,
            "path": dst,
            "gt_segment": seg,
            "note": note,
        })
    return out


# --------------------------------------------------------------------------
# 负样本
# --------------------------------------------------------------------------


def make_negative(idx: int, tmp: Path) -> Path | None:
    """生成与所有参考片无关的负样本。C3 要求必须存在。"""
    dst = tmp / f"neg{idx}.mp4"
    freq = 500 + 111 * idx
    if idx % 2 == 0:
        vf = "testsrc2=size=640x360:rate=30"
    else:
        img = tmp / f"negbase{idx}.png"
        make_base_image(100 + idx * 7).transpose(Image.Transpose.ROTATE_90).save(img)
        vf = None
    if vf:
        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", f"{vf}:duration={DUR}",
               "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={DUR}",
               "-t", str(DUR), "-c:v", "libopenh264", "-b:v", "900k",
               "-c:a", "aac", "-b:a", "96k", "-shortest", str(dst)]
    else:
        img = tmp / f"negbase{idx}.png"
        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
               "-loop", "1", "-i", str(img),
               "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={DUR}",
               "-t", str(DUR), "-vf", f"scale={W}:{H},format=yuv420p",
               "-c:v", "libopenh264", "-b:v", "900k",
               "-c:a", "aac", "-b:a", "96k", "-shortest", str(dst)]
    ok, err = _run(cmd)
    if not ok:
        print(f"    [!] 负样本 neg{idx} 失败: {err}")
        return None
    return dst


# --------------------------------------------------------------------------
# 损坏样本
# --------------------------------------------------------------------------


def make_corrupt(src: Path, dst: Path, mode: str) -> bool:
    """文件级损坏。§8.3 要求标注类型（不同位置后果差异极大）。"""
    data = bytearray(src.read_bytes())
    n = len(data)
    if n < 4096:
        return False
    if mode == "flip_byte":
        data[n // 2] ^= 0xFF
    elif mode == "flip_bits":
        rng = np.random.RandomState(SEED)
        for _ in range(64):
            i = int(rng.randint(2048, n))
            data[i] ^= 1 << int(rng.randint(0, 8))
    elif mode == "break_header":
        # 破坏 ftyp/moov 头部区域
        for i in range(16, min(64, n)):
            data[i] = 0x00
    elif mode == "truncate_10":
        data = data[: int(n * 0.95)]
    elif mode == "truncate_30":
        data = data[: int(n * 0.85)]
    elif mode == "zero_middle":
        mid = n // 2
        data[mid : mid + 8192] = b"\x00" * 8192
    else:
        return False
    dst.write_bytes(bytes(data))
    return True


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 videofp 合成评测语料")
    ap.add_argument("--out", default="data/corpus", help="输出目录")
    ap.add_argument("--refs", type=int, default=4, help="参考源片数量")
    ap.add_argument("--negs", type=int, default=4, help="负样本数量")
    ap.add_argument("--variants-per-ref", type=int, default=6, help="每条源片抽样多少个变体")
    ap.add_argument("--keep-tmp", action="store_true", help="保留中间产物")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    src_dir = root / "sources"
    neg_dir = root / "negatives"
    q_dir = root / "queries"
    tmp = root / "tmp"
    for d in (src_dir, neg_dir, q_dir, tmp):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("1/5 生成参考源片")
    print("=" * 68)
    sources: list[Path] = []
    for i in range(args.refs):
        p = src_dir / f"s{i+1}.mp4"
        if p.exists() and _probe_duration(p) > 0:
            print(f"  [skip] {p.name} 已存在")
        elif not make_source(i, p):
            continue
        sources.append(p)
        print(f"  [ok] {p.name}  {_probe_duration(p):.2f}s")
    if not sources:
        print("!! 没有可用参考源片，中止")
        return 1

    print()
    print("=" * 68)
    print("2/5 生成变体（每条源片抽样）")
    print("=" * 68)
    records: list[dict] = []  # 待检片真值
    all_variants: list[tuple[Path, list[dict]]] = []
    for i, s in enumerate(sources):
        tag = f"s{i+1}"
        vs = make_variants(s, tmp, tag)
        all_variants.append((s, vs))
        print(f"  {tag}: {len(vs)} 个变体")

    print()
    print("=" * 68)
    print("3/5 生成负样本")
    print("=" * 68)
    negs: list[Path] = []
    for i in range(args.negs):
        p = make_negative(i, tmp)
        if p:
            negs.append(p)
    print(f"  {len(negs)} 个负样本")

    print()
    print("=" * 68)
    print("4/5 中性化重命名 -> queries/qNNN（C2：文件名不含标签）")
    print("=" * 68)
    rng = np.random.RandomState(SEED)
    pool: list[dict] = []
    for (s, vs) in all_variants:
        sel = vs if args.variants_per_ref >= len(vs) else list(rng.choice(len(vs), args.variants_per_ref, replace=False))
        for k in sel:
            v = vs[int(k)]
            pool.append({
                "src": s, "variant": v["name"], "path": v["path"],
                "gt_segment": v["gt_segment"], "note": v["note"],
            })
    for p in negs:
        pool.append({"src": None, "variant": "negative", "path": p, "gt_segment": None, "note": "负样本"})

    order = rng.permutation(len(pool))
    qid_n = 0
    for oi in order:
        item = pool[int(oi)]
        qid_n += 1
        qid = f"q{qid_n:03d}"
        ext = item["path"].suffix
        dst = q_dir / f"{qid}{ext}"
        shutil.copyfile(item["path"], dst)
        records.append({
            "qid": qid,
            "file": dst.name,
            "source": "" if item["src"] is None else item["src"].name,
            "transform": item["variant"],
            "verdict": "negative" if item["src"] is None else "positive",
            "gt_segment": "" if item["gt_segment"] is None else f"{item['gt_segment'][0]}-{item['gt_segment'][1]}",
            "corrupt": "",
            "note": item["note"],
        })
    print(f"  {len(records)} 条待检片写入 {q_dir}")

    print()
    print("=" * 68)
    print("5/5 生成损坏样本 + 写真值 + 校验")
    print("=" * 68)
    # 损坏样本基于 q001 的副本
    base_q = next((r for r in records if r["verdict"] == "positive"), None)
    corrupt_rows: list[dict] = []
    if base_q:
        base_path = q_dir / base_q["file"]
        modes = ["flip_byte", "flip_bits", "break_header", "truncate_10", "truncate_30", "zero_middle"]
        for k, mode in enumerate(modes):
            qid = f"q9{k:02d}"
            d = q_dir / f"{qid}{base_path.suffix}"
            if make_corrupt(base_path, d, mode):
                corrupt_rows.append({
                    "qid": qid, "file": d.name,
                    "source": base_q["source"], "transform": f"corrupt_{mode}",
                    "verdict": "positive", "gt_segment": base_q["gt_segment"],
                    "corrupt": mode, "note": f"基于 {base_q['qid']} 的 {mode} 损坏",
                })
        print(f"  {len(corrupt_rows)} 个损坏样本")

    # --- 写 groundtruth.csv ---
    cols = ["qid", "file", "source", "transform", "verdict", "gt_segment", "corrupt"]
    with open(root / "groundtruth.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["note"])
        w.writeheader()
        for r in records:
            w.writerow(r)
    with open(root / "groundtruth_corrupt.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["note"])
        w.writeheader()
        for r in corrupt_rows:
            w.writerow(r)
    # 人类可读清单：**不参与评测**（C2）
    with open(root / "manifest_private.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "file", "source", "transform", "note"])
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in ["qid", "file", "source", "transform", "note"]})

    # --- 校验（§10.5 强制要求）---
    # 口径说明：损坏样本的**目的**就是让 ffprobe/ffmpeg 失败或降级，所以对它们
    # 不做"时长正常"校验；只对其余样本强制校验文件数与时长。
    # 否则我们会把"损坏生效"误报成"语料生成失败"。
    print()
    problems: list[str] = []
    expected_bad: list[str] = []
    for r in records:
        p = q_dir / r["file"]
        if not p.exists():
            problems.append(f"缺失文件: {p.name}")
            continue
        if _probe_duration(p) <= 0:
            problems.append(f"时长异常: {p.name}")
    for r in corrupt_rows:
        p = q_dir / r["file"]
        if not p.exists():
            problems.append(f"缺失文件（损坏样本）: {p.name}")
            continue
        if _probe_duration(p) <= 0:
            expected_bad.append(f"{p.name}({r['corrupt']})")
    for s in sources:
        if _probe_duration(s) <= 0:
            problems.append(f"源片时长异常: {s.name}")

    summary = {
        "sources": len(sources),
        "queries": len(records),
        "positives": sum(1 for r in records if r["verdict"] == "positive"),
        "negatives": sum(1 for r in records if r["verdict"] == "negative"),
        "corrupt": len(corrupt_rows),
        "corrupt_already_unprobeable": expected_bad,
        "problems": problems,
    }
    (root / "corpus_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 68)
    print("校验结果")
    print("=" * 68)
    print(f"  参考源片 {summary['sources']} | 待检 {summary['queries']} "
          f"(正 {summary['positives']} / 负 {summary['negatives']}) | 损坏 {summary['corrupt']}")
    if expected_bad:
        print(f"  说明: {len(expected_bad)} 个损坏样本已无法探测时长（预期行为，非缺陷）: "
              f"{', '.join(expected_bad)}")
    if problems:
        print(f"  !! {len(problems)} 个问题:")
        for p in problems[:10]:
            print(f"     - {p}")
        return 1
    print("  [ok] 无问题：文件数与时长校验通过")

    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)  # §10.2: 不用 rm -rf
    print(f"\n语料位于 {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
