# 现有加速开关与调参面盘点（wayfinder 票 #3）

- 调查时间：2026-10-31 13:54–14:05 CST
- 仓库：`/home/acgmn/github/WanGP`（WanGP v18.888），**只读**：未改代码、未跑生成、未提交 git
- 本机：RTX 4080 Laptop **12GB / sm89**，Python 3.13.16，torch **2.16.0.dev20261019+cu134**（CUDA 13.4），triton **3.9.0**
- 判定口径：**「可用」= 本机有直接证据**（实跑探针 / 代码分支实测命中）；**「未验证」= 门槛看起来满足但没实跑**；**「不可用」= 有反向证据**（未安装 / 硬件门槛不满足）

## 结论（TL;DR）

1. **默认姿态是「几乎全关」**：`compile=disabled`、CUDA Graphs 需模型自己声明 `cuda_graphs`、Sage 系一个都没装。`wgp_config.json` 实测值见 §7。
2. **本机真正在跑的性能路径**只有三条：`sdpa` 注意力（`auto` 解析结果）、`INT8` 量化 + **Comfy Kitchen** 内核（实测 `int8 auto -> kitchen`）、`vmm_spill` + `mmgp` 分配器。
3. **最大的未挖开关是注意力**：`sage2` 官方标 ~40% 加速且 sm89 门槛满足，但包未装；且 `docs/INSTALLATION.md:179-187` 的预编译轮子绑定 **cp311 + torch2.10 + cu130**，与本机 py3.13 + torch2.16 nightly 不匹配 → 只能源码编译，风险高。
4. `sage3` 在本机**即使装了也不可用**：`shared/attention.py:303-322` 在 `major<10` 时直接把它从可选集移除。
5. **CUDA Graph 与 attention 互斥**：图白名单 `wgp.py:3690` 只有 `sdpa/flash/sage/sage2/radial/sol`；选 `veda/vdn/vsa` 会让图直接关闭（`docs/MODELS.md:64` 同口径）。
6. 文档滞后一处：`docs/CLI.md` 仍演示 `--teacache`，但 `shared/cli_args.py` 里**没有**该参数（也无 `--lowvram/--medvram`）。

---

## 1. 总表：开关 → 作用层 → 默认值 → 本机可用性

> 作用层标注：**显存**（峰值/占用）、**速度**（吞吐/延迟）、**稳定性**（OOM、精度、可复现）；可叠加。

### 1.1 注意力（attention）

| 开关 | 作用层 | 默认值 | 出处 | 本机 sm89 判定 |
| --- | --- | --- | --- | --- |
| `attention_mode=auto` | 速度 | `auto`，按 `sage2 > sage > sdpa` 取第一个可用 | `wgp.py:2848`；`shared/attention.py:334-338` | **可用**（本机解析为 `sdpa`） |
| `attention_mode=sdpa` | 稳定性基线 | 非默认，但为实际生效值 | `shared/attention.py:276-293` | **可用**（torch 原生） |
| `sage` | 速度（标称 ~30%） | 未启用（包未装） | `shared/attention.py:276-293`；`docs/CLI.md:405-445` | **不可用**：`sageattention` 未安装（ModuleNotFoundError） |
| `sage2` | 速度（标称 ~40%，Auto 首选） | 未启用（包未装） | `shared/sage2_core.py:89-95`（只要求 major>=8） | **不可用**：包未装；硬件门槛本身满足 |
| `sage3` | 速度（标称 >50%） | 未启用 | `shared/attention.py:303-322`（`major<10` 移除） | **不可用**：sm89 被硬性剔除，装了也不能选 |
| `flash` | 速度 | 未启用 | `shared/attention.py:276-293` | **不可用**：`flash_attn` 未安装 |
| `xformers` | 速度 | 未启用 | 同上 | **不可用**：`xformers` 未安装 |
| `veda`（模型级 `attention` 覆盖） | 速度 | 仅 `*_veda` 变体自动注入 | `models/minimax_h3/minimax_h3_handler.py:729-739`；`wgp.py:6870-6884`；`defaults/minimax_h3_fl2va_veda.json:10,18` | **未验证**：门槛 `triton + CC>=8.0` 满足（`shared/attention.py:32-46`），未实跑 |
| `sol`（稀疏注意力） | 速度 | H3 变体 `custom_attention_modes` 内 | `minimax_h3_handler.py:489-491`；`shared/attention.py:32-46` | **未验证**：要求 Triton + RTX30xx+，两者满足 |
| `vsa` / `vdn` | 速度 | H3 fasth3/vdn 变体注入 | `minimax_h3_handler.py:729-739` | **未验证**：走 Triton，未实跑 |
| `radial` | 速度 | 仅 `sageattention` 2.x 且 `--betatest` 出现 | `shared/attention.py:276-293`；`wgp.py:12249-12257` | **不可用**：依赖 sageattention |
| `attention_sparsity` | 速度 / 质量 | `1.0`（H3 变体 `1.3`） | `models/_settings.json:98`；`shared/extra_settings.py:553`；`minimax_h3_handler.py:1394-1412` | **未验证**：仅对声明 `supports_sparsity` 的模式（sol/vsa 等）生效 |
| `attention_head_split` | 显存 ↓（长视频）/速度 −≤10% | `0`（Off） | `wgp.py:2860`；`shared/attention_kit.py:38-40` | **未验证**：纯 torch 分块逻辑，无硬件门槛，未实跑 |
| `--attention` CLI | 配置入口 | `""`（不改） | `shared/cli_args.py:77`；`wgp.py:3657-3664` | **可用**，但只接受 `auto/sdpa/sage/sage2/flash/xformers`——**选不到 `veda/sol/vsa/sage3/radial`**，这些只能走 `overrides` 或模型定义 |

### 1.2 编译与 CUDA Graphs

| 开关 | 作用层 | 默认值 | 出处 | 本机 sm89 判定 |
| --- | --- | --- | --- | --- |
| `compilation`（三选一：`disabled`/`cudagraph`/`transformer`） | 速度 | **`disabled`** | `wgp.py:3671-3678`；`wgp.py:2849-2854`；`docs/SETTINGS.md:143` | **可用**（两条分支均实测，见 §7） |
| `transformer`（torch.compile / inductor） | 速度（官方标 ≤10%） | 关 | `mmgp/offload.py:6699`（`torch.compile(..., backend="inductor", mode=compile_mode)`） | **可用**：`torch.compile` 实测成功（2.7 s 编译） |
| `cudagraph`（CUDA Graph 捕获） | 速度（短步长） | 关；且需 `model_def["cuda_graphs"]` 为真 | `wgp.py:4564-4568`（缺省打印「CUDA Graphs are not supported for this Model」） | **可用**：`torch.cuda.CUDAGraph` 捕获实测成功 |
| CUDA Graph 门禁（attention 白名单） | 稳定性 | `sdpa/flash/sage/sage2/radial/sol` | `wgp.py:3690-3694`；门禁键 `wgp.py:3687` | **可用**，但本机只对 `sdpa` 放行 |
| CUDA Graph 权重类型校验 | 稳定性 | 未验证的权重类型直接 `disabled` | `wgp.py:3699-3707`；`mmgp/cuda_graphs.py:1-30,44,90-99` | **未验证**：取决于 int8 权重类型是否在 `CUDA_GRAPH_WEIGHT_TYPES` |
| LLM 引擎内建 CUDA Graph | 速度 | **恒开**，无设置项 | `shared/llm_engines/lmgp/engine/model_runner.py:237-238`；`shared/llm_engines/cudagraph_kit.py:178-184`；`docs/PROMPTS.md:537` | **可用**（与 `compile` 设置无关） |
| `--compile` CLI | 配置入口 | 不传 = 读配置 | `shared/cli_args.py:75`；`wgp.py:3675-3677`（传了强制 `transformer` 并锁 UI） | **可用** |
| MPS 保护 | 稳定性 | 非 CUDA 平台强制 `disabled` | `wgp.py:3678` | 不适用 |

### 1.3 量化与精度

| 开关 | 作用层 | 默认值 | 出处 | 本机 sm89 判定 |
| --- | --- | --- | --- | --- |
| `transformer_quantization=int8` | **显存 ↓↓** | **`int8`（默认）** | `wgp.py:2849-2851`；`wgp.py:3649,3672` | **可用**（实测生效） |
| `transformer_quantization=fp8` | 显存 ↓↓ / 速度 | 非默认 | `wgp.py:3672` | **未验证**：`_scaled_mm_available(e4m3)` 实测 **True**，但未在模型上跑 |
| `transformer_quantization=bf16` | 质量（显存 ↑↑） | 非默认 | `wgp.py:3672` | **未验证**；12 GB 下 Wan2.2/H3 大概率 OOM |
| `text_encoder_quantization`（`int8`/`bf16`） | RAM / 显存 | **`int8`** | `wgp.py:2851,3673,3656` | **可用** |
| `int8_kernels`（`auto`/`wgpkernels`/`kitchen`/`triton`/`disabled`） | 速度 / 精度 | **`auto`** → 本机解析为 **`kitchen`** | `shared/kernels/int8_backend.py:24-26`；实测 `int8 auto -> kitchen` | **可用**：`comfy_kitchen` 已装并通过 `_probe_kitchen`（`int8_backend.py:76-91` 会实跑 decode+GEMM） |
| `fp8_kernels`（`auto`/`pytorch`/`triton`/`cuda`） | 速度 / 精度 | **`auto`** → 本机 `cuda` | `shared/kernels/fp8_backend.py:19-20,35-43` | **可用**：`fp8 auto -> ('cuda','cuda')` 实测 |
| `kernel_precision`（`strict`/`fast`） | 速度 / 精度 | **`fast`** | `shared/kernels/kernel_policy.py:11-12` | **可用**（进 CUDA graph 门禁键，`wgp.py:3687`） |
| `mixed_precision`（`0`=16bit only / `1`=16/32 混合） | 显存 / 质量 | **`"0"`** | `wgp.py:4425-4427,4488`；`plugins/configuration/plugin.py:412` | **可用** |
| `transformer_dtype_policy`（Auto/`fp16`/`bf16`，`--fp16/--bf16`） | 精度 / 显存 | **`""`（Auto）** | `wgp.py:3651-3655,3305-3323` | **可用** |
| `boost`（1=ON / 2=OFF，CFG>1 时层卸载提速） | 速度（~10%）/ 显存 +1 GB | **`1`（ON）** | `wgp.py:2856`；`plugins/configuration/plugin.py:415` | **未验证**：需 CFG>1 场景实测 |
| `--save-quantized` | 加载速度 / 磁盘 | 关 | `shared/cli_args.py:30` | **未验证** |
| `--convrot` | 速度 / 精度 | 关 | `shared/cli_args.py:31` | **未验证** |
| `vae_config` / VAE 精度 | 显存 | `0`；VAE 恒 float16 | `wgp.py:2849-2860`；`wgp.py:4487` | **可用**（16-bit VAE 为硬编码注释「VAEs always run in 16-bit」） |

### 1.4 显存 / 分配器 / 进程级

| 开关 | 作用层 | 默认值 | 出处 | 本机 sm89 判定 |
| --- | --- | --- | --- | --- |
| `--profile`（0/1/2/3/3.5/4/4.5/5） | **显存**（分档） | **4**（LowRAM_LowVRAM，推荐） | `wgp.py:12235-12242`；`shared/cli_args.py:41`；`wgp.py:2794`（`float()`，故 4.5 可用） | **可用**（实测配置 `video_profile: 4`） |
| `--profile 4.5` | 显存 ↓ ~1 GB / 速度 略降 | 可选 | 同上 | **未验证** |
| `--vram-allocator`（`default`/`vmm`/`vmm_spill`） | **显存**（碎片/spill） | **`vmm_spill`** | `shared/cuda_memory.py:96-97`；`shared/cli_args.py:62`；库 `mmgp/allocator/vmm_alloc_linux_x86_64.so` | **可用**（间接：配置已持久化且应用多次启动过；未单独压测 spill） |
| `--ram-allocator`（`default`/`mmgp`） | 内存（释放即归还系统） | **`mmgp`** | `shared/cuda_memory.py:19,100-109`（失败自动降级） | **可用**（间接，同上） |
| `--perc-reserved-mem-max` | 内存（pinning 上限） | `0`（→ 配置值，Windows 0.4 / Linux 0.6） | `shared/cli_args.py:55` | **可用** |
| `--preload` | 显存 / 首次出片 | `"0"` MB | `shared/cli_args.py:33` | **未验证** |
| `--vram-safety-coefficient` | **稳定性**（OOM 余量） | `0.85` | `shared/cli_args.py:26` | **可用** |
| `--max-vram` / `--max-ram` / `--ram-stress` | 显存 / 内存（虚拟卡、压力测试） | 未设 | `shared/cli_args.py:59-61`；`shared/cuda_memory.py:96-107` | **未验证** |
| `PYTORCH_CUDA_ALLOC_CONF`（`expandable_segments:True`） | 显存碎片 | **仅 Linux 且 `mode=expandable` 时写**；默认 `vmm` 不走该分支 | `mmgp/allocator/__init__.py:43,47-49,60-69` | **未验证**：当前 `vmm_spill` 未命中该分支 |
| `CUDA_CACHE_PATH` / `CUDA_CACHE_MAXSIZE`（4 GiB） | 首次编译速度 | 自动设置，4 GiB | `shared/cuda_memory.py:23,83` | **可用** |
| `TORCHINDUCTOR_CACHE_DIR` | 编译缓存 | 随 int8 后端自动指向 `<inductor>/wangp_int8_v{N}/<backend>` | `shared/kernels/int8_backend.py:49-67`（写入在 `:65`） | **可用** |
| `TRITON_DEFAULT_FP_FUSION=1` | 速度 | `1`（`setdefault`） | `wgp.py:11` | **可用**（Triton 3.9 实测可跑 kernel） |
| `WANGP_CUDA_STACK_BYTES` | 稳定性（CUDA 栈） | 未设 → 库内 256 | `shared/cuda_memory.py:22,24,156` | **未验证** |
| `--prevent-power-throttling` | 稳定性（仅 Windows） | 开 | `shared/cli_args.py:63` | 不适用（本机 Linux/WSL） |

### 1.5 步数 / 引导 / 采样（速度的直接旋钮）

| 开关 | 作用层 | 默认值 | 出处 | 本机 sm89 判定 |
| --- | --- | --- | --- | --- |
| `num_inference_steps` | **速度**（线性） | 全局 `30`；H3 handler 覆盖为 `20` | `models/_settings.json:31`；`minimax_h3_handler.py:1394-1412` | **可用** |
| `num_inference_steps2` | 速度 | `3` | `models/_settings.json:32`；`minimax_h3_handler.py:1394-1412` | **可用** |
| `guidance_phases` | 速度 / 质量 | `1`；Wan2.2 多子模型时 `2` | `models/_settings.json:35`；`wgp.py:11571`；`models/wan/wan_handler.py:1444-1445`；`defaults/t2v_2_2.json:19` | **可用** |
| `flow_shift` | 质量 / 收敛 | `3`（全局兜底另算 5/7） | `models/_settings.json:51`；`wgp.py:3505`；`defaults/t2v_2_2.json:23`（12） | **可用**（纯采样参数） |
| `skip_steps_multiplier` / `skip_steps_start_step_perc` | 速度 | `1.75` / `0` | `models/_settings.json:63-64` | **未验证**；Wan2.2 走 `no_steps_skipping=True`（`wan_handler.py:197-199`） |
| `--steps` | 配置入口 | `0`（不改） | `shared/cli_args.py:47`；`wgp.py:3535-3537` | **可用** |

---

## 2. `defaults/*.json` 面（要求 1）

### 2.1 这些文件的形状

`defaults/*.json` 只有两层：`{"model": {...}}` + 一堆**顶层生成参数**。加载在 `wgp.py:3590-3613`：先 `model_def = json_def.pop("model")`，再把剩下的顶层参数塞回 `model_def["settings"] = json_def`（`wgp.py:3613`）。

优先级链（`wgp.py:3506-3510`）：`models/_settings.json` 的 primary_settings → handler 的 `update_default_settings` → defaults 文件的 `settings` 覆盖（`ui_defaults.update(copy.deepcopy(model_settings))`）。

### 2.2 例 1：Wan 2.2 —— `defaults/t2v_2_2.json`

| 键 | 值 | 行 |
| --- | --- | --- |
| `guidance_phases` | `2` | `defaults/t2v_2_2.json:19` |
| `switch_threshold` | `875` | `defaults/t2v_2_2.json:20` |
| `guidance_scale` | `4` | `defaults/t2v_2_2.json:21` |
| `guidance2_scale` | `3` | `defaults/t2v_2_2.json:22` |
| `flow_shift` | `12` | `defaults/t2v_2_2.json:23` |
| `num_inference_steps` | **文件里没有**，落到 `models/_settings.json:31` 的 `30` | — |
| attention 相关 | **文件里没有**；该模型 `URLs2` 触发 `models/wan/wan_handler.py:197-199`（`no_steps_skipping=True`、`compile=["transformer","transformer2"]`） | — |

### 2.3 例 2：MiniMax H3 —— `defaults/minimax_h3_fl2va.json` 与 veda 变体

`defaults/minimax_h3_fl2va.json` **全文只有 model + prompt**，11 行，**没有任何** attention / guidance / flow_shift / steps 字段 —— H3 的默认值全部来自 handler（`models/minimax_h3/minimax_h3_handler.py:1394-1412`）：

| 参数 | H3 通用分支默认值 | 行 |
| --- | --- | --- |
| `video_length` | `124` | `:1394-1412` |
| `sliding_window_size` / `overlap` | `362` / `18` | `:1394-1412` |
| `num_inference_steps` | `20`（`ptd` 变体 8；fasth3 走 `FASTH3_STEPS`） | `:1394-1412`、`:1422`、`:1424` |
| `guidance_phases` / `num_inference_steps2` | `1` / `3` | `:1394-1412` |
| `guidance_scale` | `1.0` | `:1394-1412` |
| `flow_shift` | `12.0`（viggle 变体 3.0，`3` 步） | `:1394-1412`、`:1379` |
| `sample_solver` | `euler` | `:1394-1412` |
| `attention_sparsity` | `1.3` | `:1394-1412` |
| `skip_steps_start_step_perc` / `skip_steps_multiplier` | `25` / `0.08` | `:1394-1412` |

**盘上实测印证**：`settings/minimax_h3_fl2va_settings.json` 里 `:33 num_inference_steps=20`、`:36 guidance_phases=1`、`:38 switch_threshold=0.9035`、`:40 guidance_scale=1.0`、`:52 flow_shift=12.0` —— 与 handler 默认值一致（`settings/t2v_2_2_settings.json` 不存在，说明本机还没用 Wan2.2 生成过）。

**唯一的 attention 结构化入口**是 veda 变体（全仓库只有 4 个 defaults 带 `"veda": true`）：
`defaults/minimax_h3_fl2va_veda.json:10 "veda": true`，`:13-15` modules 指向 `minimax_h3_t2va_veda_predictor_*`，`:18 "overrides": "attention_mode='veda'"`。同构：`minimax_h3_fl2va_pruned_veda.json`、`minimax_h3_ref2va_veda.json`、`minimax_h3_ref2va_pruned_veda.json`。

### 2.4 defaults 的统计事实

- 290 个 `defaults/*.json`；`grep -l 'sdpa|sage|flash|veda|attention'` 只命中 **22 个**，且几乎全是 description 文本或 `"sample_solver": "flash"` 之类的误命中 —— **defaults 层几乎不控制注意力**。
- 带 `guidance_phases` 的 35 个文件里：`1` 出现 17 次、`2` 出现 16 次、`3` 出现 1 次、`0` 出现 1 次。
- `flow_shift`：`5`（26 次）最多，其次 `10.0`（9）、`12/12.0`（5）、`2`（5）、`3`（14）。
- **defaults/*.json 里没有任何 `"steps"` 键**（`grep '"steps"'` 空）—— 步数一律由全局或 handler 决定。

---

## 3. `torch.compile` / CUDA Graphs（要求 2）

**总开关**：`compilation`，三档 `disabled` / `cudagraph` / `transformer`（`wgp.py:3674`），默认 **`disabled`**（`wgp.py:3671`、`wgp.py:2849-2854`）。

- **`transformer`（torch.compile）**：`wgp.py:4559` 计算 `compile_modules`，最终在 `mmgp/offload.py:6699` 执行 `submodel.forward = torch.compile(submodel.forward, backend="inductor", mode=compile_mode)`；`compile_mode` 默认 `"default"`（`mmgp/offload.py:6231,6252`）。官方标称 ≤10% 提速、编译很慢（`wgp.py:3674` 的 UI 文案）。
- **`cudagraph`**：`wgp.py:4564-4568` 读 `model_def["cuda_graphs"]`；缺省时打印「CUDA Graphs are not supported for this Model」，并置 `kwargs["cudaGraphs"]`；`wgp.py:4580-4581` 再跑 `check_cuda_graph_weights`。
- **门禁与键**：`wgp.py:3687` 注册图键（attention / head_split / kernel_precision / int8 与 fp8 后端 revision）；`wgp.py:3690` 白名单 `{"sdpa","flash","sage","sage2","radial","sol"}`；`wgp.py:3693-3694` 不在集合内直接报「the attention mode '...' is not supported by them」。**所以 veda/vsa/vdn 与 CUDA Graph 互斥**，与 `docs/MODELS.md:64` 一致。
- **自动捕获**：`mmgp/cuda_graphs.py:1-30`（按 weights/LoRA/shapes 生成 key，首次 eager、之后 capture），`:44 MAX_KEYS=8`，`:90-99 scope`，`:120-128 is_enabled`。
- **哪些模型进图**：`models/minimax_h3/transformer.py:468/758/954` 无条件 `with cuda_graphs.scope():`；`models/wan/modules/model.py` 里没有 `cuda_graphs` 引用（Wan 视频塔走 mmgp 的 block 调度）。
- **LLM 引擎自带 CUDA graph，不受该设置影响**：`shared/llm_engines/lmgp/engine/model_runner.py:237-238`、`shared/llm_engines/cudagraph_kit.py:178-184`、`docs/PROMPTS.md:537`（「with no setting to choose」）。
- 改 `profile`/`compilation`/`quantization` 会**重载模型**（`docs/SETTINGS.md:143`）。

---

## 4. 量化与精度（要求 3）

选择面全部是**全局配置 + 单次生成 overrides**，`defaults/*.json` 里没有任何 `quantize`/`qtype`/`mixed_precision`（grep 空）。

1. **Transformer 权重三档**（`wgp.py:3672`）：`Scaled Int8`（默认）/`Scaled Fp8`/`16-bit (bf16)`。读取默认值在 `wgp.py:3649`。
2. **文本编码器两档**（`wgp.py:3673`）：`16-bit (bf16)`、`8-bit (int8，默认)`（`wgp.py:3656`）。
3. **数值精度策略**：`transformer_dtype_policy` 默认 `""`（Auto），`--fp16`/`--bf16` 直接改写（`wgp.py:3651-3655`），解析见 `get_transformer_dtype`（`wgp.py:3305-3323`）。`mixed_precision` 默认 `"0"`=纯 16-bit，`"1"`=16/32 混合（`wgp.py:4425-4427,4488`）。VAE 恒 16-bit（`wgp.py:4487`）。
4. **内核后端**：`int8_kernels`（`shared/kernels/int8_backend.py:24-26`，auto 排序 wgpkernels → kitchen → triton → pytorch）、`fp8_kernels`（`shared/kernels/fp8_backend.py:19-20`，auto 优先 CUDA FP8 tensor core）、`kernel_precision`（`shared/kernels/kernel_policy.py:11-12`，默认 `fast`）。
5. **单次生成可覆盖**五类：`memory_profile`/`compilation`/`attention_mode`/`transformer_quantization`/`text_encoder_quantization`（`shared/utils/generation_overrides.py:7`，校验在 `wgp.py:12281-12282`，生效在 `wgp.py:4197`）。
6. UI 入口：`plugins/configuration/plugin.py:410-419`（量化、dtype、mixed precision、compile、boost、int8/fp8 内核、kernel precision、head split 全在这一段）。

---

## 5. CLI 参数（要求 4）

argparse **只存在于 `shared/cli_args.py`（108 行）**，`wgp.py` 自己没有 parser。性能相关：

| 参数 | 默认 | 行 | 备注 |
| --- | --- | --- | --- |
| `--profile` | `-1` | `shared/cli_args.py:41` | `wgp.py:2794 float()`，支持 `4.5`；档位含义见 `wgp.py:12235-12242` |
| `--compile` | 不传 | `:75` | 传了强制 `transformer` 并锁 UI（`wgp.py:3675-3677`） |
| `--attention` | `""` | `:77` | 只接受 6 档（`wgp.py:3659`） |
| `--steps` | `0` | `:47` | 覆盖 `num_inference_steps`（`wgp.py:3535-3537`） |
| `--fp16` / `--bf16` | 不传（Auto） | `:51` / `:52` | |
| `--perc-reserved-mem-max` | `0` | `:55` | |
| `--preload` | `"0"` MB | `:33` | |
| `--vram-allocator` | `vmm_spill` | `:62` | 候选 `default`/`vmm`/`vmm_spill` |
| `--ram-allocator` | `mmgp` | `:58` | 候选 `default`/`mmgp` |
| `--vram-safety-coefficient` | `0.85` | `:26` | |
| `--max-vram` / `--max-ram` / `--ram-stress` | 未设 | `:59` / `:60` / `:61` | 虚拟卡 / 压力测试 |
| `--save-quantized` / `--convrot` | 关 | `:30` / `:31` | |
| `--vram-debug MIN_MB` / `--ram-debug` | 关 | `:56` / `:57` | 出报告到 `<outputs>/vram_debug`、`ram_debug` |
| `--prevent-power-throttling` | 开（仅 Windows） | `:63` | BooleanOptionalAction |
| `--test` / `--verbose` / `--advanced` / `--gpu` / `--frames` | — | `:32` / `:42` / `:50` / `:65` / `:48` | |

**不存在**：`--lowvram`、`--medvram`、`--teacache`、`--fast`（`grep` 两者皆空）。显存档位只有 `--profile`。`docs/CLI.md` 仍演示 `--teacache`，属文档滞后。

---

## 6. 环境变量入口（要求 5）

| 变量 | 设定位置 | 值 / 行为 |
| --- | --- | --- |
| `TRITON_DEFAULT_FP_FUSION` | `wgp.py:11` | `setdefault "1"` |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `wgp.py:13` | `"1"`（AMD） |
| `OPENBLAS_NUM_THREADS` | `wgp.py:15` | `min(8, cpu_count)` |
| `NUMBA_THREADING_LAYER` | `wgp.py:35-36` | Linux → `workqueue` |
| `HF_HUB_OFFLINE` | `wgp.py:16-17` | `--offline` 时置 1 |
| `GRADIO_LANG` | `wgp.py:9` | 默认 `en` |
| `CUDA_CACHE_PATH` / `CUDA_CACHE_MAXSIZE` | `shared/cuda_memory.py:83` | 缓存 4 GiB（`:23`） |
| `WANGP_CUDA_STACK_BYTES` | `shared/cuda_memory.py:22,156` | 覆盖 CUDA 栈预留（默认 256，`:24`） |
| `PYTORCH_CUDA_ALLOC_CONF` | `mmgp/allocator/__init__.py:60-69` | **仅 `mode=="expandable"` 且 Linux** 才追加 `expandable_segments:True`；`install()` 必须在 CUDA 初始化前（`:47-49`） |
| `TORCHINDUCTOR_CACHE_DIR` | `shared/kernels/int8_backend.py:65` | 指向 `<inductor>/wangp_int8_v{N}/<backend>`，并 `torch.compiler.reset()` |
| `WAN2GP_QUANTO_INT8_KERNEL` | `shared/kernels/int8_backend.py:939` | `"1"`/`"0"` 切 Quanto 内核 |
| `WAN2GP_FP8_TRITON_CACHE` / `_AUTOTUNE` / `_MAX_SLOTS` / `_DECODE` | `shared/kernels/fp8_w8a16_triton.py:28,39,40,144` | `_DECODE` 在 CC >= (8,9) 走原生路径 |
| `WAN2GP_NVFP4_*` | `shared/kernels/nvfp4_w4a16_triton.py:26,35,36` | 同构 |
| `TORCH_COMPILE` / `TORCHINDUCTOR` / `PYTORCH_ENABLE_MPS_FALLBACK` | `shared/mps/device_patch.py:15-17` | 仅 MPS，置 0 |
| `MMGP_VIRTUAL_VRAM_GB` / `MMGP_VIRTUAL_RAM_GB` | `mmgp/virtual_gpu.py` | 虚拟显存/内存（等价 `--max-vram/--max-ram`） |
| `WANGP_DEBUG_UI` | `wgp.py:2797` | UI 调试 |

**未找到**：仓库内任何 `TORCHINDUCTOR_*` 策略开关（除上面的缓存目录）、`TRITON_CACHE_DIR`、`TRITON_HOME`。

---

## 7. 本机可用性证据（实跑，非推测）

### 7.1 已持久化的生效配置（`wgp_config.json`，应用自己写的）

```
:2  "attention_mode": "auto"        :8  "compile": "disabled"
:4  "transformer_quantization": "int8"   :10 "boost": 1
:5  "text_encoder_quantization": "int8"  :11 "int8_kernels": "auto"
:13 "kernel_precision": "fast"      :12 "fp8_kernels": "auto"
:14 "attention_head_split": 0       :25 "video_profile": 4
:2375 "perc_reserved_mem_max": 0    :2376 "vram_allocator": "vmm_spill"
:2377 "ram_allocator": "mmgp"       :2378 "mixed_precision": "0"
:2379 "transformer_dtype_policy": ""
```

### 7.2 探针结果

| 探针 | 结果 |
| --- | --- |
| `torch` / `triton` 版本 | `2.16.0.dev20261019+cu134` / `3.9.0`，`cap (8,9)`，设备 `NVIDIA GeForce RTX 4080 Laptop GPU` |
| `torch.compile(dynamic=False)` bf16 小模型 | `torch.compile OK in 2.7s` |
| `torch.cuda.CUDAGraph` 捕获 + replay | `CUDAGraph capture OK` |
| 手写 Triton kernel（add, BLOCK=256） | `triton 3.9.0 OK True` |
| `scaled_fp8._scaled_mm_available(float8_e4m3fn)` | **True**（`_init_scaled_mm_support()` 后） |
| `fp8_backend.resolve("auto")` | `('cuda','cuda')` —— CUDA FP8 tensor core 路径命中 |
| `int8_backend.resolve("auto")` | `'kitchen'` —— Comfy Kitchen 内核**实跑**过 decode + GEMM 才被选中 |
| `wgpkernels` 可导入？ | **否**（`ModuleNotFoundError`），故 auto 落到 kitchen |
| `comfy_kitchen` 可导入？ | **是** |
| `sageattention` / `flash_attn` / `xformers` | **全部缺失**（`ModuleNotFoundError`） |
| `optimum.quanto` | 已装（int8 量化基础） |

---

## 8. 未验证项与风险（给下一张票）

1. **Sage 系（sage/sage2）是最可能的免费提速**，但预编译轮子（`docs/INSTALLATION.md:179-187`，cp311 + torch 2.10 + cu130）与本机（py3.13 + torch2.16 nightly + cu134）不匹配；源码编译需要 CUDA 13 工具链 + 头文件，风险：ABI、编译期 OOM、kernel 崩溃。
2. **veda / sol / vsa / vdn 全部未验证**：静态门槛（Triton + CC 8.0+）全满足，但需实跑一次 MiniMax H3（`*_veda`、fasth3/vdn 变体）才能写入「可用」。
3. **fp8 量化未在模型上跑过**：`scaled_mm` 支持已实证，但 `transformer_quantization=fp8` 的端到端正确性/显存收益未测。
4. **`--profile 4.5`、`--preload`、`boost`、`attention_head_split`、`skip_steps_*`** 均未实跑；其中 `attention_head_split` 对 12 GB + 长视频是最对症的显存旋钮（`plugins/configuration/plugin.py:419` 称「much lower VRAM for long videos, up to 10% slower」）。
5. **`vmm_spill` 的 spill 行为未压测**：本次只证明它已是持久化配置、应用能启动，未证明在 12 GB 边界上是否真的 spill 而非 OOM（可用 `--vram-debug` 出报告验证）。
6. **文档滞后**：`docs/CLI.md` 的 `--teacache` 已不存在于 `shared/cli_args.py`；`docs/CLI.md:424` 仍建议 RTX20XX 装 SageAttention 1.0.6。

---

## 9. 关键出处索引

- 注意力选择：`shared/attention.py:18-46`（门槛）、`:276-293`（可选集）、`:303-322`（按架构剔除）、`:334-338`（默认序 sage2>sage>sdpa）、`:351-360`（resolve）、`:458`（pay_attention）；`shared/sage2_core.py:89-95`
- 注意力生效链：`wgp.py:6870-6884`（模型级 `attention` 覆盖）、`wgp.py:7649-7671`（三级优先 + 写 `_attention`）、`wgp.py:3657-3664`（CLI）
- 编译/图：`wgp.py:3671-3678`、`wgp.py:4559-4581`、`wgp.py:3687-3707`、`mmgp/offload.py:6699`、`mmgp/cuda_graphs.py:1-30,44,90-99`
- 量化/内核：`wgp.py:3649-3673`、`shared/kernels/int8_backend.py:24-26,76-91`、`shared/kernels/fp8_backend.py:19-20,35-43`、`shared/kernels/kernel_policy.py:11-12`、`shared/attention_kit.py:38-40`
- 显存/分配：`shared/cuda_memory.py:19-24,83,96-124`、`mmgp/allocator/__init__.py:43-69`
- 默认值：`models/_settings.json:31-98`、`models/minimax_h3/minimax_h3_handler.py:1379-1424`、`models/wan/wan_handler.py:197-199,1444-1445`、`defaults/t2v_2_2.json:19-23`、`defaults/minimax_h3_fl2va_veda.json:10,18`
- CLI：`shared/cli_args.py`（全文 108 行）
- 文档口径：`docs/SETTINGS.md:143`、`docs/MODELS.md:64`、`docs/PROMPTS.md:537`、`docs/CLI.md:405-445`、`docs/INSTALLATION.md:179-187`
