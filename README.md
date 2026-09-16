# phashVerify — 视频盗版/抄袭鉴定原型（pHash 感知指纹）

输入一个疑似盗版视频与一个参考片库，输出：**是否命中**、**命中哪个参考片**、
**命中的时间码区间**（可作证据）、以及判定依据（匹配帧数、覆盖率、平均 Hamming
距离、疑似倍速系数）。

> 本项目是 [Agent.md](Agent.md) 所述 `videofp` 目标的一个可运行实现。
> 原交接文档记录的是另一台机器（Linux / Pentium G4560 / 无 GPU），
> **本机环境与原文不符**，已在 Agent.md §0 与 §3 逐条修正。

## 硬约束

| 编号 | 约束 |
|---|---|
| C1 | **禁止用视觉能力判断两个视频是否同一份**。所有判定只能来自算法输出的数字 |
| C2 | 评测不泄漏标签：待检文件名中性化（`q001.mp4`），真实含义只写在 `groundtruth.csv` |
| C3 | 必须包含负样本，否则假阳率无法计算 |
| C7 | 输出可复现：同输入 → 同数字 |
| C8 | 不在测试集上反复调阈值；阈值扫描必须离线用已存分数重算 |

## 管线

```
视频文件
  → 解码（ffmpeg rawvideo 管道，含 MPEG-TS 重封装降级链）
  → 几何归一化（黑边裁剪 + 方阵化）
  → pHash64 + 镜像哈希
  → 序列对齐 + 片段定位（对角线投票 + 倍速搜索 + 稳健精修）
  → 音频地标指纹多尺度匹配
  → 融合打分 + 时间码证据
```

## 环境

- Windows 10 + PowerShell 7（另需 `ffmpeg`/`ffprobe`）
- Python 3.13（由 `uv` 管理虚拟环境）
- 依赖：`numpy`、`Pillow`、`scipy`；`imagehash` **仅用于测试对照**

`ffmpeg` 不在 PATH 时，`vfp.decode` 会按以下顺序查找：仓库内 `ffmpeg/bin/` →
环境变量 `VFP_FFMPEG_DIR` → 系统 PATH。

```powershell
# 建虚拟环境并装依赖
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe numpy pillow imagehash scipy
```

## 快速开始

```powershell
# 1) 正确性测试（17 项）
.venv\Scripts\python.exe -m tests.test_core

# 2) 端到端演示：自建语料 → 建库 → 查询 → 打印时间码证据
.venv\Scripts\python.exe demo.py

# 3) 正式语料 + 建库 + 评测
.venv\Scripts\python.exe -m tools.make_corpus --out data/corpus --refs 3 --negs 3 --variants-per-ref 5
.venv\Scripts\python.exe -m vfp.cli build --sources "data/corpus/sources/*.mp4" --db db
.venv\Scripts\python.exe -m vfp.cli eval  --db db --gt data/corpus/groundtruth.csv --report results/full.json --sweep

# 4) 单条查询
.venv\Scripts\python.exe -m vfp.cli query --db db --query data/corpus/queries/q001.mp4
```

## 实测结果

合成语料（3 参考片 / 15 正 / 3 负），2 fps，Hamming 阈值 ≤10，启用黑边裁剪 +
镜像双查询 + 音频：

```
recall 1.000 | false_positive_rate 0.000 | precision 1.000 | accuracy 1.000
source_top1_acc_on_positives 1.000 | segment_f1_tau50 1.000
```

⚠️ **口径限制**：仅 3 个负样本（FPR 分辨率上限 1/3）、合成素材、
`mean_iou` 仅 2 条有真值。**这些数字是"管线跑通"的证据，不是论文结果。**
详见 [Agent.md](Agent.md) §14。

## 目录结构

```
vfp/
  hashes.py      pHash64 / dct256 / dHash64 / 镜像哈希 / 帧质量分
  decode.py      ffmpeg 定位、ffprobe 探测、容错解码
  normalize.py   黑边检测裁剪、方阵归一化
  align.py       对角线投票对齐、片段定位、倍速精修、片段 IoU/F1
  audiofp.py     谱图峰值地标指纹、多尺度变速、偏移投票
  index.py       Config、建库、端到端匹配、融合打分
  eval.py        评测指标、离线阈值扫描
  cli.py         build / query / eval
tools/make_corpus.py   合成语料生成 + 真值 + 自校验
tests/test_core.py     17 项正确性测试
demo.py                端到端演示
ffmpeg/bin/            随仓库携带的 ffmpeg（未纳入版本管理）
```

## 已知限制（诚实清单）

1. **损坏视频无法修复**：破坏容器头/截断的样本在解复用阶段即失败
   （`moov atom not found`），降级链无法救回，只能如实报告"解码失败"。
2. **短查询的倍速不可辨识**：候选倍速命中数并列时不输出精修倍速，
   避免把时间码算错。
3. **实现为线性扫描**：参考片规模上去需要 ANN 索引。
4. **全部结论基于合成语料**，真实影视语料尚未接入。
5. `chroma` 音频通道未实现（原交接文档中已判死，本机未复核）。

## 许可与来源

随仓库携带的 `ffmpeg/bin/` 来自
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)（LGPL 构建），
许可证见 `ffmpeg/LICENSE.txt`。该目录未纳入版本管理。
