"""ffmpeg / ffprobe 定位、探测与容错解码。

设计约束：
  - 通过 ``rawvideo`` + 管道直出灰度帧，**不落盘 PNG**（磁盘与 IO 友好）
  - Windows 下 ffmpeg 不在 PATH，优先使用随仓库携带的 ``ffmpeg/bin/``
  - 解码失败时走降级链（见 ``decode_gray``），并如实回报 ``decode_note``

``decode_gray`` 返回的 ``DecodeResult`` 会带上 ``note``，调用方需要在报告里
原样输出，禁止把降级过的结果冒充正常解码。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "ffmpeg_path",
    "ffprobe_path",
    "probe",
    "decode_gray",
    "DecodeResult",
    "ProbeInfo",
]

# Windows: 避免弹出控制台窗口
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# 仓库根 = 本文件上两级
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _candidates(name: str) -> list[Path]:
    exe = f"{name}.exe" if os.name == "nt" else name
    cands = [
        _REPO_ROOT / "ffmpeg" / "bin" / exe,  # 随仓库携带（本项目实际使用）
        Path(os.environ.get("VFP_FFMPEG_DIR", "")) / exe if os.environ.get("VFP_FFMPEG_DIR") else None,
    ]
    return [c for c in cands if c is not None]


def _resolve(name: str) -> str:
    """定位 ffmpeg/ffprobe 可执行文件；找不到则抛 FileNotFoundError。"""
    for c in _candidates(name):
        if c.is_file():
            return str(c)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"未找到 {name}。请把 ffmpeg 放到 {_REPO_ROOT / 'ffmpeg' / 'bin'}，"
        f"或设置环境变量 VFP_FFMPEG_DIR，或加入 PATH。"
    )


def ffmpeg_path() -> str:
    return _resolve("ffmpeg")


def ffprobe_path() -> str:
    return _resolve("ffprobe")


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_NO_WINDOW,
        timeout=timeout,
    )


# --------------------------------------------------------------------------
# 探测
# --------------------------------------------------------------------------


@dataclass
class ProbeInfo:
    path: str
    ok: bool
    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: float = 0.0
    vcodec: str = ""
    acodec: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "ok": self.ok,
            "width": self.width,
            "height": self.height,
            "duration": round(self.duration, 4),
            "fps": round(self.fps, 4),
            "vcodec": self.vcodec,
            "acodec": self.acodec,
            "error": self.error,
        }


def probe(path: str | Path) -> ProbeInfo:
    """用 ffprobe 读取容器/流信息。失败时 ``ok=False`` 且 ``error`` 有内容。"""
    path = str(path)
    cmd = [
        ffprobe_path(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    info = ProbeInfo(path=path, ok=False)
    try:
        p = _run(cmd, timeout=120)
    except Exception as e:  # noqa: BLE001
        info.error = f"ffprobe 调用失败: {e}"
        return info
    if p.returncode != 0:
        info.error = (p.stderr.decode("utf-8", "replace") or "ffprobe 返回非零").strip()[:500]
        return info
    try:
        data = json.loads(p.stdout.decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        info.error = f"JSON 解析失败: {e}"
        return info

    fmt = data.get("format", {}) or {}
    info.duration = float(fmt.get("duration") or 0.0)
    for st in data.get("streams", []) or []:
        kind = st.get("codec_type")
        if kind == "video" and not info.vcodec:
            info.vcodec = st.get("codec_name", "") or ""
            info.width = int(st.get("width") or 0)
            info.height = int(st.get("height") or 0)
            # 优先用 avg_frame_rate，回退 r_frame_rate
            for key in ("avg_frame_rate", "r_frame_rate"):
                fr = st.get(key) or ""
                if "/" in fr:
                    num, den = fr.split("/")[:2]
                    try:
                        den_f = float(den)
                        if den_f:
                            info.fps = float(num) / den_f
                            break
                    except ValueError:
                        pass
            if not info.duration:
                try:
                    info.duration = float(st.get("duration") or 0.0)
                except (TypeError, ValueError):
                    pass
        elif kind == "audio" and not info.acodec:
            info.acodec = st.get("codec_name", "") or ""

    info.ok = bool(info.width and info.height)
    if not info.ok:
        info.error = info.error or "未找到可用视频流"
    return info


# --------------------------------------------------------------------------
# 解码
# --------------------------------------------------------------------------


@dataclass
class DecodeResult:
    """解码结果。``note`` 必须如实反映是否走了降级路径。"""

    ok: bool
    frames: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), np.uint8))
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    expected_frames: int = 0
    note: str = ""
    error: str = ""
    recovered: bool = False

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def usable_ratio(self) -> float:
        if self.expected_frames <= 0:
            return 1.0 if self.n_frames else 0.0
        return min(1.0, self.n_frames / self.expected_frames)


def _scale_expr(width: int, height: int, target_w: int | None) -> str:
    """生成保持宽高比、宽高取偶数的 scale 表达式。"""
    if target_w is None:
        # 原尺寸，仅确保偶数
        return f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
    return f"scale={target_w}:trunc(ih*{target_w}/iw/2)*2"


def _decode_once(
    path: str,
    fps: float,
    target_w: int | None,
    max_frames: int | None,
    timeout: int,
) -> tuple[bool, bytes, str, int, int]:
    """跑一次 ffmpeg 解码。返回 (ok, raw_bytes, err, width, height)。"""
    info = probe(path)
    vf = f"fps={fps},{_scale_expr(info.width, info.height, target_w)}"
    cmd = [
        ffmpeg_path(),
        "-hide_banner", "-loglevel", "error",
        "-nostdin",
        "-i", path,
        "-an", "-sn", "-dn",
        "-vf", vf,
        "-pix_fmt", "gray",
        "-f", "rawvideo",
        "-",
    ]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_NO_WINDOW,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, b"", "解码超时", 0, 0
    except Exception as e:  # noqa: BLE001
        return False, b"", f"ffmpeg 调用失败: {e}", 0, 0

    if p.returncode != 0 or not p.stdout:
        err = (p.stderr.decode("utf-8", "replace") or "").strip().splitlines()
        return False, b"", (err[-1] if err else "ffmpeg 无输出"), 0, 0

    # 反推输出尺寸：rawvideo 无头部，用字节数 / 帧数推断
    # 先按目标宽算高度；若 target_w 为 None 用探测尺寸
    if target_w is None:
        w, h = info.width, info.height
    else:
        w = target_w
        h = max(2, int(round(info.height * target_w / info.width / 2.0)) * 2) if info.width else 0
    if w <= 0 or h <= 0:
        return False, b"", "无法推断输出尺寸", 0, 0
    frame_bytes = w * h
    n = len(p.stdout) // frame_bytes
    if n == 0:
        return False, b"", "解码得到 0 帧", 0, 0
    if max_frames:
        n = min(n, max_frames)
    return True, p.stdout[: n * frame_bytes], "", w, h


def _repackage_mpegts(src: str, dst: str) -> tuple[bool, str]:
    """降级步骤 1：重封装为 MPEG-TS，绕过损坏的容器头/索引。"""
    cmd = [
        ffmpeg_path(),
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-fflags", "+discardcorrupt+genpts",
        "-err_detect", "ignore_err",
        "-i", src,
        "-c", "copy",
        "-f", "mpegts",
        "-y", dst,
    ]
    p = _run(cmd, timeout=300)
    if p.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True, ""
    err = (p.stderr.decode("utf-8", "replace") or "").strip().splitlines()
    return False, (err[-1] if err else "重封装失败")


def decode_gray(
    path: str | Path,
    fps: float = 2.0,
    target_w: int | None = 320,
    max_frames: int | None = None,
    timeout: int = 600,
) -> DecodeResult:
    """解码为灰度帧序列 ``(N, H, W)`` uint8。

    降级链：
      1. 直接解码（容错 flag 开启）
      2. 重封装为 MPEG-TS 后再解码
    每一步都会记录到 ``note``，``recovered`` 标记是否用过降级路径。
    """
    path = str(path)
    if not os.path.isfile(path):
        return DecodeResult(ok=False, error=f"文件不存在: {path}")

    info = probe(path)
    dur = info.duration or 0.0
    expected = int(round(dur * fps)) if dur > 0 else 0

    # --- 尝试 1: 直接解码 ---
    ok, raw, err, w, h = _decode_once(path, fps, target_w, max_frames, timeout)
    note = "正常解码"

    # --- 尝试 2: MPEG-TS 重封装兜底 ---
    if not ok:
        tmp_dir = _REPO_ROOT / "data" / "corpus" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ts = tmp_dir / (Path(path).stem + ".recover.ts")
        rep_ok, rep_err = _repackage_mpegts(path, str(ts))
        if rep_ok:
            ok2, raw2, err2, w2, h2 = _decode_once(str(ts), fps, target_w, max_frames, timeout)
            try:
                ts.unlink()
            except OSError:
                pass
            if ok2:
                ok, raw, err, w, h = ok2, raw2, err2, w2, h2
                note = "降级：mpegts 重封装后解码成功"
            else:
                err = f"{err} | 重封装后仍失败: {err2}"
        else:
            err = f"{err} | 重封装失败: {rep_err}"

    if not ok:
        return DecodeResult(
            ok=False, fps=fps, expected_frames=expected,
            note="解码失败", error=err, duration=dur,
        )

    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).copy()
    return DecodeResult(
        ok=True,
        frames=frames,
        fps=fps,
        width=w,
        height=h,
        duration=dur,
        expected_frames=expected,
        note=note,
        recovered=note.startswith("降级"),
    )
