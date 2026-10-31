# 可上游化项识别（#8）

判定口径：对 12–16 GB 显卡用户普遍有用、且改动局部、不改变默认行为的，才建议上游；依赖本机环境（.wslconfig、自编译 wheel）的一律不提。

## 候选清单

### U1 H3 的 VAE tiling 参数被忽略（建议提，收益最高）
- 证据：`models/minimax_h3/pipeline.py:522-523`
  ```python
  def _configure_tiling(self, _tile_size):
      self.vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256)
  ```
  形参被丢；调用点 `pipeline.py:810` 传的是 `VAE_tile_size`，而它来自 `wgp.py:7674-7685` 按 `vae_config` 与显存算出的 `get_VAE_tile_size`（`models/minimax_h3/video_vae.py:58-61`，0→256、2→512、3→256）。
- 影响：UI 里 VAE tiling 的 Default / Low VRAM 两档对 H3 完全无效，12 GB 卡想靠 tile 降显存没有任何手段。
- 上游已有相关工作：无（`git log -S "_configure_tiling"` 只有引入提交）；Wan 系的同名字段是生效的（`models/wan/modules/vae.py:1394-1395`）。
- 最小 diff（~6 行）：
  ```python
  def _configure_tiling(self, tile_size):
      if tile_size == 0:                      # vae_config == 1：关闭 tiling
          self.vae.use_tiling = False
          return
      t = tile_size or 256
      self.vae.enable_tiling(tile_sample_min_height=t, tile_sample_min_width=t)
  ```
- 风险：tile=512 在 24 GB 卡上更省时间、tile=256 在低显存更省显存，需两档都测一遍画质（tile 边界有重叠混合）。

### U2 显存预算表被整体替换，缺少按模型覆盖入口（建议提，中等）
- 证据：`wgp.py:4253-4258` 把 mmgp 的 profile 预算改成 `transformer=100MB / text_encoder=100MB / text_encoder_2=100MB / *=3000MB`（profile 5 时 1000）。
- 影响：12 GB 卡上 `*` 的 3000 MB 与实际 VAE 三件套（2.62+0.6+0.64 GB）不匹配，差额靠 `vmm_spill`（`shared/cuda_memory.py:34`、`:132-146`）溢到内存，表现为变慢而非 OOM；用户无法只调某一类模型的预算。
- 现有可用手段：`shared/utils/generation_overrides.py` 能改 `memory_profile`，但改不了单项预算。
- 最小 diff：把 `defaults` 抽成 `DEFAULT_BUDGETS` 常量，并从 `server_config.get("budgets", {})` 合并（现成 `kwargs.setdefault` 已有，只需在覆盖前先并入用户字典），不改变默认值。
- 通用性：高（所有 8–16 GB 用户），但属于「给专家开的旋钮」，建议作为可选 PR，说明默认行为不变。

### U3 预取窗口硬编码为 None（建议提，中等）
- 证据：`mmgp/offload.py:6848-6849` `asyncTransfers=True, prefetchWindow=None` → 只有 window+1 = 2 个 slot（每 block ≈385 MB ≈0.77 GB）；`offload.py:4229-4232` 支持 `prefetch_window=2` → 3 slot ≈1.15 GB。
- 影响：12 GB 卡上多花 0.38 GB 显存换更少搬运等待，是可调的性价比旋钮，但目前没有配置入口。
- 最小 diff：`prefetchWindow` 读 `server_config.get("prefetch_window")`，默认仍 None。
- 风险：显存<12 GB 的用户开 2 可能 OOM，所以必须默认关。

### U4 attention 模式缺少 GUI 入口（建议提，低）
- 证据：`shared/utils/generation_overrides.py:7` 已支持 `attention_mode='sage2'` 字符串覆盖，但设置面板没有对应键（`wgp.py:12281` 的 `choices` 里只有 memory_profile / compilation / attention_mode 的 override 组合，不是会话设置）。
- 影响：用户要逐次对比 sdpa / sage2 / veda，只能改 CLI 或重启。
- 最小 diff：把 `attention_mode` 加进 `wgp_config.json` 的会话键并在启动时读取（已有 `get_attention_override_choices`）。

### U5 pyproject / uv 支持（建议提，范围大）
- 现状：仓库只有 `setup.py` + `requirements.txt`；本机已写出可用的 `pyproject.toml` + `uv.lock`（332 包），但里面含本机 wheel 路径（taichi、sageattention），上游版需去掉这些。
- 最小可上游范围：只加 `pyproject.toml`（`[tool.uv] package=false` + 依赖列表 + 索引），不加 lock，不加本机 path source。
- 通用性：高，但维护成本在上游，属于「提议而非 PR」。

## 不建议上游（本机特调）
- `.wslconfig` 的 40 GB / 16 GB swap：纯本机 Windows 设置。
- sage2 源码编译（c++20 + sm89 + cu13.3）：`setup.py` 的 `-std=c++17` 问题是上游 SageAttention 的事，WanGP 侧只需文档说明。
- mmgp 的 torch 版本区间放宽：属于本机对 mmgp vendored 代码的临时改动，应先看上游 mmgp 是否已支持 2.16。
- profile 4 的 pinnedMemory 行为：属于 mmgp 设计取舍，不作为 PR。

## 建议动作
先提 U1（bug 级别、6 行、有明确用户价值）；U2/U3 合并成一个「暴露预算与预取旋钮」的 PR；U4 单独小 PR；U5 先开 issue 征询维护者意见再动。

## 落地状态（2026-10-31 更新）

- **U1 已提 PR**：[deepbeepmeep/WanGP#2569](https://github.com/deepbeepmeep/WanGP/pull/2569)（`fix/minimax-h3-vae-tiling`，6 行；本地实测 720p/49f：256=376.0 s、512=363.6 s、关闭=364.8 s，峰值 6593/6330/6214 MiB，差异在噪声内但旋钮现在真的生效）。
- **U3 已在上游**：`wgp.py:4572` 已有 `if "prefetch_window" in server_config: kwargs["prefetchWindow"] = ...`，本条作废（调研时按 `mmgp/offload.py:6848` 判断，漏看了这层转发）。
- **U2 已提 PR**：[deepbeepmeep/WanGP#2570](https://github.com/deepbeepmeep/WanGP/pull/2570)（`feat/configurable-vram-budgets`，读 `wgp_config.json` 的 `budgets` 覆盖表；优先级 handler > 用户 > profile 默认，默认行为不变）。
- **U4 已在上游**：设置页本来就有 `Attention Type` 下拉（`plugins/configuration/plugin.py:406-408`，`choices=attention_modes_choices`），`server_config["attention_mode"]` 也会持久化；本条作废。
- **U5 未动**：建议先开 issue 问维护者是否接受 `pyproject.toml`（不含 lock、不含本机 path source）。
