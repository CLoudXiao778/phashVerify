# CHANGELOG

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 结构。

## [0.2.0] — 本次接手会话

在本机（Windows 10 / i5-12400F / 无 CUDA）从零实现 `videofp` 目标的可运行管线。
接手时仓库内只有 `Agent.md` 与两个 0 字节文件，**没有任何代码**。

### Added

- `vfp/hashes.py` —— pHash64（与 `imagehash.phash` 位级一致）、`dct256`、
  `dHash64`、帧质量分（Laplacian 方差）、popcount 查表版 Hamming 矩阵
- `vfp/decode.py` —— ffmpeg/ffprobe 自动定位、ffprobe JSON 探测、
  `rawvideo` 管道直出灰度帧（不落盘 PNG），含 MPEG-TS 重封装降级链
- `vfp/normalize.py` —— 基于逐像素时间最小值的黑边检测与裁剪、方阵归一化
- `vfp/align.py` —— 对角线偏移投票、`topk` 候选裁剪、片段聚类与定位、
  倍速网格搜索 + 多尺度分层稳健精修 + **可辨识性判据**、片段 IoU 与 τ=0.5 的 P/R/F1
- `vfp/audiofp.py` —— STFT 峰值地标指纹、组合哈希、时间偏移直方图投票、
  多尺度变速回放（0.95/1.0/1.05）
- `vfp/index.py` —— `Config`（含宽容反序列化与 `SCHEMA_VERSION`）、建库、
  端到端查询、视觉/音频融合打分、时间码证据输出
- `vfp/eval.py` —— recall / FPR / precision / accuracy / top1 / mean IoU /
  片段 F1 / usable_ratio，以及**离线**阈值扫描
- `vfp/cli.py` —— `build` / `query` / `eval` 子命令，含消融开关
- `tools/make_corpus.py` —— 合成语料生成（本地绘制素材，无下载）、
  文件名中性化、真值写出、**生成后文件数与时长自校验**
- `tests/test_core.py` —— 17 项正确性测试
- `demo.py` —— 端到端演示（镜像 / 黑边 / 干净副本 / 字节翻转 / 破坏容器头 / 负样本）
- `README.md`、`CHANGELOG.md` —— 原文为 0 字节，本次写入

### Fixed（相对交接文档的描述修正的真实缺陷）

- **镜像双查询**：原文档只说"翻转哈希参与比对"。实测证明哈希位重排**数学上
  不成立**（镜像使 DCT 奇数列系数翻转符号），置换后距离 34 bit ≈ 随机。
  改为在像素域翻转帧后重算哈希，命中率 0/16 → 16/16。
- **倍速估计过拟合**：短查询（12 帧/6 秒）下 1.0 与 1.05 命中数并列，精修把
  倍速拉到 0.8457，导致时间码被算错（6 秒报成 5 秒）。新增可辨识性判据。
- **对齐偏移搜索写错**：初版按"参考帧下标"而非"offset"分组投票，等价于假设
  偏移恒为 0，任何倍速都返回同一偏移。改为按 offset 直方图投票。
- **DCT 尺度不一致**：自研正交 DCT 与 `scipy.fftpack.dct` 尺度不同，4 张图共
  112 bit 判定不一致。改用 scipy 后差异 0 bit。
- **音频通道整条静默失效**：`ndarray == "miss"` 返回数组，`if` 判断抛
  "truth value of an array is ambiguous"，被 `except` 吞掉。改用 `in` 判键。
- **负样本也报出源片**：全部候选 0 分时仍把第一个参考片写成 `source`，
  会误导读报告的人。改为仅在 `n_hits > 0` 时报源片。
- **语料生成滤镜名错误**：本机 ffmpeg 构建**没有 `eq` 滤镜**，也没有
  `brightness` 滤镜；`colorlevels` 选项名是 `rimin` 而非 `rim`。
  改用 `colorlevels=rimin=...` + `huesaturation`。
- **损伤样本校验误报**：损坏样本"探测不到时长"是预期行为，却被当成生成失败。
  改为对损坏样本不做时长校验并单独列出。

### Environment

- 安装随仓库携带的 ffmpeg 8.1.2（BtbN win64 LGPL 静态构建）到 `ffmpeg/bin/`；
  该构建**无 libx264/libx265、无 `eq` 滤镜**
- 用 `uv` 创建 Python 3.13 虚拟环境，安装 numpy / Pillow / imagehash / scipy
- 初始化 git 仓库，配置远程 `origin`
  `https://github.com/CLoudXiao778/phashVerify.git`

### Known limitations

- 破坏容器头/截断的损坏样本在解复用阶段即失败（`moov atom not found`），
  降级链无法救回，仅能如实报告解码失败
- 参考片匹配为线性扫描，规模上去需要 ANN
- 全部结论基于合成语料，真实影视语料尚未接入
- `chroma` 通道未实现（原文档已判死，本机未复核）
- 消融实验、哈希选型对比、语义检索层均未做

### Verified

- 17/17 正确性测试通过
- 全量评测 18 条：recall 1.000 / FPR 0.000 / top1 1.000 / 片段 F1 1.000
- 损坏样本评测 6 条：3 条优雅降级命中、3 条解复用失败
- 详见 `Agent.md` §14（含口径警告与未完成清单）
