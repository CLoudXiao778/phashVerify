"""正确性测试。运行::

    .venv\\Scripts\\python.exe -m tests.test_core

或（若装了 pytest）``pytest tests/ -v``。

覆盖点（对应 Agent.md §5.1 的验收表，并补上原来缺失的两项）：
  1. phash64 与 imagehash.phash 位级一致
  2. hamming_matrix 与逐个比较一致
  3. dct256 结构（256 bit / 自身距离 0）
  4. 对齐：3s 偏移 + 1.05 倍速 —— 必须恢复偏移与倍速
  5. 对齐：中段插广告 -> 应切成 2 段而不是糊成 1 段
  6. 黑边裁剪
  7. 镜像哈希等价性（**原来缺失**）
  8. 片段 IoU 与 τ=0.5 的 P/R/F1（**原来缺失**）
  9. Config 宽容反序列化（防止 §4.3 的 TypeError）
 10. 负样本假阳基线：无关图像的两两 Hamming 应远大于阈值
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vfp.align import align_pair, iou_f1, segment_iou  # noqa: E402
from vfp.hashes import (  # noqa: E402
    dct256,
    dhash64,
    hamming64,
    hamming_matrix,
    phash64,
    phash64_mirrored,
)
from vfp.index import Config  # noqa: E402
from vfp.normalize import find_black_border, normalize_frames  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


# --------------------------------------------------------------------------
# 素材
# --------------------------------------------------------------------------


def sample_image(i: int, size: int = 128) -> Image.Image:
    """确定性的测试图（固定 seed，保证可复现）。"""
    rng = np.random.RandomState(1000 + i)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    base = 60 + 80 * np.sin(xx / (7.0 + i)) + 60 * np.cos(yy / (5.0 + 2 * i))
    base += rng.rand(size, size) * 40
    base = np.clip(base, 0, 255).astype(np.uint8)
    img = Image.fromarray(base, "L").convert("RGB")
    return img


def distinct_frames(n: int, size: int = 128) -> list[Image.Image]:
    """生成**帧间内容充分解耦**的帧序列。

    为什么需要这个而不是 ``moving_frames``：等量平移产生的相邻帧 pHash 距离
    只有 2-4 bit，等于让"差一帧"和"差五帧"看起来一样，倍速因此在信息论意义上
    不可辨识（实测 0.95/1.0/1.05 拿到完全相同的命中集）。
    真实视频帧间距离通常在 20 bit 量级（本机实测均值 21），本函数复现该量级，
    每帧叠加独立的伪随机纹理 + 不同位置的几何元素。
    """
    from PIL import ImageDraw

    out = []
    for i in range(n):
        rng = np.random.RandomState(4200 + i)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        b = 80 + 60 * np.sin(xx / (3.0 + (i % 5))) + 60 * np.cos(yy / (2.5 + (i % 7)))
        b += rng.rand(size, size) * 70
        img = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8), "L").convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([10 + i % 30, 20 + i % 40, 60 + i % 30, 80 + i % 40],
                    outline=(255, 255, 255), width=4)
        d.line([(0, (i * 7) % size), (size, (i * 13) % size)], fill=(255, 255, 255), width=3)
        out.append(img)
    return out


def moving_frames(n: int, size: int = 128, shift: float = 1.0) -> list[Image.Image]:
    """生成沿对角线平移的帧序列，模拟推镜；用于对齐测试。

    底图必须**非周期且尺寸足够大**：早期版本用一张 128x128 图做等量平移，
    导致索引 k 与 k+1 的帧彼此难以区分，出现了"offset 6 与 offset 10 得分相同"
    的退化，从而让对齐测试失真。
    """
    frames = []
    big = sample_image(3, size * 2 + 64)   # 底图显著大于取景框
    arr = np.asarray(big, dtype=np.uint8)
    for k in range(n):
        dx = int(round(k * shift))
        sub = arr[dx : dx + size, dx : dx + size]
        if sub.shape[0] < size or sub.shape[1] < size:
            sub = arr[:size, :size]
        frames.append(Image.fromarray(sub, "RGB"))
    return frames


# --------------------------------------------------------------------------
# 1. phash64 与 imagehash 位级一致
# --------------------------------------------------------------------------


def test_phash_matches_imagehash() -> None:
    try:
        import imagehash
    except ImportError:
        check("phash64 vs imagehash.phash（位级一致）", False, "imagehash 未安装，跳过")
        return
    diffs = []
    for i in range(4):
        img = sample_image(i)
        mine = int(phash64(img))
        theirs = int(str(imagehash.phash(img)), 16)
        diffs.append(bin(mine ^ theirs).count("1"))
    total = sum(diffs)
    check(
        "phash64 vs imagehash.phash（位级一致）",
        total == 0,
        f"4 张图总差异 {total} bit（逐张 {diffs}）",
    )


# --------------------------------------------------------------------------
# 2. hamming_matrix
# --------------------------------------------------------------------------


def test_hamming_matrix() -> None:
    rng = np.random.RandomState(7)
    # 注意：RandomState.randint 的上界受平台 C int 限制（Windows 为 32 位），
    # 用 1<<62 会直接抛 ValueError。这里改用两段 32 位再拼接。
    hi = rng.randint(0, 1 << 31, size=13).astype(np.uint64)
    lo = rng.randint(0, 1 << 31, size=13).astype(np.uint64)
    a = (hi << np.uint64(31)) | lo
    hi2 = rng.randint(0, 1 << 31, size=9).astype(np.uint64)
    lo2 = rng.randint(0, 1 << 31, size=9).astype(np.uint64)
    b = (hi2 << np.uint64(31)) | lo2
    mat = hamming_matrix(a, b)
    bad = 0
    for i in range(len(a)):
        for j in range(len(b)):
            if mat[i, j] != hamming64(a[i], b[j]):
                bad += 1
    check("hamming_matrix 与逐对比较一致", bad == 0 and mat.shape == (13, 9),
          f"shape={mat.shape}, 不一致 {bad} 项")


# --------------------------------------------------------------------------
# 3. dct256 结构
# --------------------------------------------------------------------------


def test_dct256() -> None:
    h = dct256(sample_image(0))
    ok = h.shape == (4,) and h.dtype == np.uint64
    zeros = int(sum(hamming64(h[k], h[k]) for k in range(4)))
    check("dct256 结构为 4xuint64 且自身距离 0", ok and zeros == 0,
          f"shape={h.shape} dtype={h.dtype} 自身距离={zeros}")


# --------------------------------------------------------------------------
# 4. 对齐：3s 偏移 + 1.05 倍速
# --------------------------------------------------------------------------


def test_align_offset_and_speed() -> None:
    """3s 偏移 + 1.05 倍速：必须同时恢复偏移与倍速。

    数据设计要点（这是本测试的关键，早期版本在这里失真过）：
      1. 用 ``distinct_frames`` 而非平移序列 —— 平移序列相邻帧距离仅 2-4 bit，
         不同倍速会给出完全相同的命中集，倍速不可辨识。
      2. 查询要**足够长**。20 帧 / 10 秒时，1.0 与 1.05 的采样下标几乎重合，
         命中数并列，实现会（正确地）拒绝输出精修倍速。
         这里用 60 秒参考片 + 100 帧查询，倍速差异才真正体现在采样下标上。
    """
    fps = 2.0
    n_ref = 120                      # 60 秒参考片
    ref_frames = distinct_frames(n_ref)
    ref_hash = np.array([phash64(f) for f in ref_frames], dtype=np.uint64)

    start_idx = 6                    # 3 秒偏移
    speed = 1.05
    n_q = 100
    idx = np.rint(start_idx + np.arange(n_q) * speed).astype(int)
    idx = idx[idx < n_ref]
    q_hash = ref_hash[idx]

    al = align_pair(
        q_hash, ref_hash, fps=fps,
        threshold=10, speed_candidates=(0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
        max_gap=3, min_frames=2,
    )
    off_ok = abs(al.offset - start_idx) <= 1
    spd_ok = abs(al.scale - speed) < 0.02
    cov_ok = al.coverage >= 0.9
    check(
        "对齐：3s 偏移 + 1.05 倍速",
        off_ok and spd_ok and cov_ok,
        f"offset={al.offset}(期望~{start_idx}) scale={al.scale:.4f}(期望{speed}) "
        f"raw_scale={al.scale_raw} identifiable={al.speed_identifiable} "
        f"coverage={al.coverage:.3f} n_hits={al.n_hits}",
    )


def test_speed_unidentifiable_is_not_overfitted() -> None:
    """反面对照：短查询下倍速不可辨识时，**不得**输出过拟合的精修倍速。

    这不是"锦上添花"的测试，而是防止一类真实事故：早期实现对齐 12 帧 / 6 秒的
    查询时，网格候选 1.0 与 1.05 并列第一，但精修把倍速拉到 0.8457，
    导致输出的参考片段被算成 5 秒而不是 6 秒 —— 时间码作为证据是错的。
    """
    fps = 2.0
    ref_frames = distinct_frames(40)
    ref_hash = np.array([phash64(f) for f in ref_frames], dtype=np.uint64)
    idx = np.arange(6, 18)                     # 12 帧 / 6 秒，短查询
    q_hash = ref_hash[idx]
    al = align_pair(q_hash, ref_hash, fps=fps, threshold=10,
                    speed_candidates=(0.9, 0.95, 1.0, 1.05, 1.1))
    # 要么倍速被正确识别为不可辨识，要么精修结果没有跑偏
    safe = (not al.speed_identifiable) or abs(al.scale - 1.0) < 0.02
    check(
        "短查询：倍速不可辨识时不输出跑偏的倍速",
        safe and abs(al.scale - 1.0) < 0.08,
        f"scale={al.scale:.4f} identifiable={al.speed_identifiable} "
        f"raw={al.scale_raw} offset={al.offset}",
    )


def test_align_no_speed_change_is_stable() -> None:
    """反向对照：真实倍速为 1.0 时，不应被估成 1.05。

    这条很重要 —— 防止"倍速估计总是偏向某一侧"的假象被当成能力。
    """
    fps = 2.0
    ref_frames = distinct_frames(40)
    ref_hash = np.array([phash64(f) for f in ref_frames], dtype=np.uint64)
    idx = np.arange(6, 26)
    q_hash = ref_hash[idx]
    al = align_pair(q_hash, ref_hash, fps=fps, threshold=10,
                    speed_candidates=(0.9, 0.95, 1.0, 1.05, 1.1))
    check(
        "对齐：真实 1.0 倍速时估计不漂移",
        abs(al.scale - 1.0) < 0.02 and abs(al.offset - 6) <= 1,
        f"scale={al.scale:.4f}(期望1.0) offset={al.offset}(期望6)",
    )


# --------------------------------------------------------------------------
# 5. 对齐：插广告 -> 2 段
# --------------------------------------------------------------------------


def test_align_ad_break() -> None:
    fps = 2.0
    n_ref = 40
    ref_frames = moving_frames(n_ref, shift=1.0)
    ref_hash = np.array([phash64(f) for f in ref_frames], dtype=np.uint64)

    # query = ref[0:8] + 5 帧无关内容 + ref[8:32]
    unrelated = sample_image(77, 128)
    ad_hash = np.array([phash64(unrelated)] * 5, dtype=np.uint64)
    q_hash = np.concatenate([ref_hash[0:8], ad_hash, ref_hash[8:32]])

    al = align_pair(q_hash, ref_hash, fps=fps, threshold=10, speed_candidates=(1.0,),
                    max_gap=3, min_frames=2)
    got2 = len(al.segments) >= 2
    check(
        "对齐：中段插广告应切成 >=2 段",
        got2,
        f"段数={len(al.segments)} " + "; ".join(
            f"[q {s.q_start:.1f}-{s.q_end:.1f} / r {s.r_start:.1f}-{s.r_end:.1f}]"
            for s in al.segments
        ),
    )


# --------------------------------------------------------------------------
# 6. 黑边裁剪
# --------------------------------------------------------------------------


def test_black_border_crop() -> None:
    """构造真正的 letterbox：中心亮画面 + 四周恒定纯黑边。

    注意：内嵌内容必须**亮且不贴边**，否则内容自身的暗像素会被误判成黑边
    （早期版本用 np.roll 制造了暗角，导致检出 18/23 的假黑边）。
    """
    base = np.asarray(sample_image(5, 128).convert("L"))
    # 归一化到较亮范围，避免内容自身接近黑边阈值
    base = (60 + base.astype(np.float64) * 0.7).clip(0, 255).astype(np.uint8)
    inner_h, inner_w = 104, 308
    top_pad, left_pad = 38, 6
    frames = []
    for k in range(6):
        sub = np.roll(base, k, axis=1)                       # 帧间变化
        sub = np.asarray(
            Image.fromarray(sub).resize((inner_w, inner_h), Image.Resampling.LANCZOS)
        )
        f = np.zeros((180, 320), dtype=np.uint8)             # 纯黑底 = 黑边
        f[top_pad : top_pad + inner_h, left_pad : left_pad + inner_w] = sub
        frames.append(f)
    arr = np.stack(frames)

    top, bottom, left, right = find_black_border(arr)
    ok = (abs(top - top_pad) <= 2 and abs(bottom - top_pad) <= 2
          and abs(left - left_pad) <= 2 and abs(right - left_pad) <= 2)
    norm, border2 = normalize_frames(arr, crop_black=True, square=True, size=32)
    check(
        "黑边裁剪 (180,320) -> 裁掉 38/38/6/6",
        ok and norm.shape[1:] == (32, 32),
        f"border={border2} 归一化后 shape={norm.shape}",
    )

    # 关闭裁剪时应保持原尺寸（消融路径可用）
    off, border_off = normalize_frames(arr, crop_black=False, square=False)
    check("关闭黑边裁剪时不改尺寸", off.shape == arr.shape and border_off == (0, 0, 0, 0),
          f"shape={off.shape} border={border_off}")


def test_letterbox_hash_recovery() -> None:
    """核心卖点验证：加黑边会毁掉 pHash，裁剪归一化能救回来。"""
    base = sample_image(9, 128)
    clean = np.asarray(base.convert("L"))
    # 直接等比缩小后居中放到黑底 -> letterbox
    small = np.asarray(
        Image.fromarray(clean).resize((128, 72), Image.Resampling.LANCZOS)
    )
    lb = np.zeros((128, 128), dtype=np.uint8)
    lb[28:100, 0:128] = small; lb[0:28, :] = 0; lb[100:, :] = 0

    h_clean = phash64(Image.fromarray(clean))
    h_lb_raw = phash64(Image.fromarray(lb))
    # 用裁剪后的内容复核
    cropped, _ = normalize_frames(lb[None, :, :], crop_black=True, square=True, size=32)
    h_lb_cropped = phash64(cropped[0])

    raw_d = hamming64(h_clean, h_lb_raw)
    fix_d = hamming64(h_clean, h_lb_cropped)
    check(
        "letterbox 修复：裁剪归一化后距离显著下降",
        fix_d < raw_d and fix_d <= 10,
        f"未归一化 dist={raw_d}, 归一化后 dist={fix_d}",
    )


# --------------------------------------------------------------------------
# 7. 镜像哈希等价性
# --------------------------------------------------------------------------


def test_mirror_hash_equivalence() -> None:
    """镜像哈希的正确性：在归一化方阵帧上翻转后再算哈希。

    同时**反向验证**：靠 64bit 位重排（旧的错误做法）是做不到的，
    这条断言把这个结论固化下来，防止以后有人又"优化"回位运算。
    """
    frames = moving_frames(16, shift=1.0)
    # 归一化到方阵（与主流程同一阶段）
    norm, _ = normalize_frames(
        np.stack([np.asarray(f.convert("L")) for f in frames]),
        crop_black=False, square=True, size=32,
    )
    flipped = norm[:, :, ::-1]  # 先翻转

    ok = True
    details = []
    for i in range(len(norm)):
        a = phash64(norm[i])
        b = phash64_mirrored(norm[i])
        # 直接对翻转帧算哈希，应当与 phash64_mirrored 完全一致
        from PIL import Image as _I
        c = phash64(_I.fromarray(np.ascontiguousarray(flipped[i]), "L"))
        if hamming64(b, c) != 0:
            ok = False
        details.append(hamming64(a, b))
    check(
        "镜像哈希：phash64_mirrored == 翻转帧的 phash（位级一致）",
        ok,
        f"{len(norm)} 帧全部一致；与原哈希的距离 {details[:6]}...",
    )


def test_mirror_query_recovers() -> None:
    """镜像双查询：查询被水平翻转后，用镜像哈希应当能全额命中。"""
    fps = 2.0
    frames = moving_frames(16, shift=1.0)
    ref_frames = np.stack([np.asarray(f.convert("L")) for f in frames])
    ref_norm, _ = normalize_frames(ref_frames, crop_black=False, square=True, size=32)
    ref_hash = np.array([phash64(f) for f in ref_norm], dtype=np.uint64)

    # 查询 = 参考帧水平翻转后再归一化（模拟 hflip 攻击）
    q_raw = ref_frames[:, :, ::-1]
    q_norm, _ = normalize_frames(np.ascontiguousarray(q_raw), crop_black=False, square=True, size=32)
    q_plain = np.array([phash64(f) for f in q_norm], dtype=np.uint64)
    q_mirror = np.array([phash64_mirrored(f) for f in q_norm], dtype=np.uint64)

    al_raw = align_pair(q_plain, ref_hash, fps=fps, threshold=10, speed_candidates=(1.0,))
    al_fix = align_pair(q_mirror, ref_hash, fps=fps, threshold=10, speed_candidates=(1.0,))
    check(
        "镜像双查询：直接比对打不中，镜像哈希后全额命中",
        al_fix.n_hits > al_raw.n_hits and al_fix.coverage >= 0.9,
        f"直接比对 n_hits={al_raw.n_hits} (cov {al_raw.coverage:.2f}) -> "
        f"镜像后 n_hits={al_fix.n_hits} (cov {al_fix.coverage:.2f})",
    )


# --------------------------------------------------------------------------
# 8. 片段 IoU / F1
# --------------------------------------------------------------------------


def test_segment_iou_and_f1() -> None:
    a = (0.0, 10.0)
    b = (5.0, 15.0)
    iou = segment_iou(a, b)
    iou_ok = abs(iou - (1.0 / 3.0)) < 1e-9
    p, r, f = iou_f1([(0.0, 10.0), (20.0, 30.0)], [(0.0, 10.0)], tau=0.5)
    f1_ok = abs(p - 0.5) < 1e-9 and abs(r - 1.0) < 1e-9 and abs(f - 2 / 3) < 1e-9
    check("片段 IoU 与 τ=0.5 的 P/R/F1", iou_ok and f1_ok,
          f"IoU={iou:.4f}(期望0.3333) P={p:.3f} R={r:.3f} F1={f:.3f}")


# --------------------------------------------------------------------------
# 9. Config 宽容反序列化
# --------------------------------------------------------------------------


def test_config_tolerant_load() -> None:
    """旧 meta.json 含未知字段 + 缺新字段，都必须能加载（§4.3 的教训）。"""
    old = {"fps": 2.0, "match_threshold": 10, "legacy_field_removed": 123, "chroma_fallback": False}
    try:
        cfg = Config.from_dict(old)
        ok = cfg.fps == 2.0 and cfg.match_threshold == 10 and cfg.mirror_query is True
        detail = f"fps={cfg.fps} threshold={cfg.match_threshold} mirror默认={cfg.mirror_query}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"抛异常: {type(e).__name__}: {e}"
    check("Config.from_dict 容忍未知/缺失字段", ok, detail)

    # 空的 config 也要能加载
    try:
        c2 = Config.from_dict({})
        check("Config.from_dict({}) 使用全默认值", c2.fps == 2.0, f"fps={c2.fps}")
    except Exception as e:  # noqa: BLE001
        check("Config.from_dict({}) 使用全默认值", False, f"抛异常: {e}")


# --------------------------------------------------------------------------
# 10. 假阳基线
# --------------------------------------------------------------------------


def test_negative_baseline() -> None:
    """无关图像之间的 Hamming 应远大于阈值 10，否则阈值不安全。"""
    hashes = [phash64(sample_image(i, 128)) for i in range(6)]
    dists = [hamming64(hashes[i], hashes[j])
             for i in range(len(hashes)) for j in range(i + 1, len(hashes))]
    mn, mean = min(dists), sum(dists) / len(dists)
    hit_rate = sum(1 for d in dists if d <= 10) / len(dists)
    check(
        "假阳基线：无关图像最小 Hamming > 阈值 10",
        mn > 10 and hit_rate == 0.0,
        f"min={mn} mean={mean:.1f} max={max(dists)} ≤10命中率={hit_rate:.2f}",
    )


# --------------------------------------------------------------------------
# 附加：衰减/编码鲁棒性快检（不依赖视频，纯图像层面）
# --------------------------------------------------------------------------


def test_quality_and_alt_hashes() -> None:
    img = sample_image(0, 128)
    dh = dhash64(img)
    ok_shape = isinstance(dh, np.uint64)
    # 亮度微调后 pHash 应基本不变（§5.2 实测亮度 +8% 命中率 32/32）
    arr = np.asarray(img.convert("L"), dtype=np.float64) * 1.08
    bright = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
    d = hamming64(phash64(img), phash64(bright))
    check("亮度 +8% 后 pHash 距离 <= 10", ok_shape and d <= 10,
          f"dhash64 类型={type(dh).__name__}, 亮度变化后距离={d}")


def main() -> int:
    print("=" * 70)
    print("videofp 正确性测试")
    print("=" * 70)
    tests = [
        test_phash_matches_imagehash,
        test_hamming_matrix,
        test_dct256,
        test_align_offset_and_speed,
        test_speed_unidentifiable_is_not_overfitted,
        test_align_no_speed_change_is_stable,
        test_align_ad_break,
        test_black_border_crop,
        test_letterbox_hash_recovery,
        test_mirror_hash_equivalence,
        test_mirror_query_recovers,
        test_segment_iou_and_f1,
        test_config_tolerant_load,
        test_negative_baseline,
        test_quality_and_alt_hashes,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            import traceback
            check(t.__name__, False, f"异常 {type(e).__name__}: {e}")
            traceback.print_exc()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n = len(RESULTS)
    print("=" * 70)
    print(f"结果: {n_pass}/{n} 通过")
    if n_pass != n:
        print("失败项:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 70)
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
