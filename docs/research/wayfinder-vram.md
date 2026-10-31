# wayfinder #2 · 12GB 卡上 WanGP 的显存去向与 offload 机制

> 调查方式：只读源码 + 本机 `nvidia-smi` / `torch.cuda` 实测，**未跑任何生成任务**。
> 所有结论后缀 `路径:行号`；标注〔推断〕的条目是源码可证但未实测的推算。
> 适用基线：RTX 4080 Laptop（12282 MiB，sm89），`wgp_config.json`：`profile=4`、`vae_config=0`、`vram_allocator=vmm_spill`、`transformer_quantization=int8`、`text_encoder_quantization=int8`、`enhancer_enabled=3`、`deepy_enabled=1`、`smart_memory_pinning=true`。

---

## 0. 一句话结论

1. **20B 权重从不整模进显存。** profile 4 下 DiT 的 50 个 block 常驻**锁页（pinned）系统内存**，denoise 时逐 block 搬进固定 slot，显存里同时只有 **2–3 个 block ≈ 0.77–1.15 GB**（slot 创建与尺寸：`mmgp/offload.py:4354-4378`）。
2. **真正吃显存的不是权重**，而是三件事：① 桌面/浏览器空载已占的 **3036 MiB**；② denoise 单步的激活/attention；③ text encoder 与 VAE 换入时的 3000 MB 级预算。
3. **mmgp 只按「总显存」算预算，不知道别的进程在占。** `self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory`（`mmgp/offload.py:4027`）→ 它规划 `0.85 × 12281 ≈ 10439 MB` 的模型占用（`:6346`、`:6545-6554`），而实际只剩 ≈ 9.2 GB → 差额靠 **VMM allocator 的 spill** 溢到系统内存（`shared/cuda_memory.py:34`、`:132-146`），表现为**变慢而不是 OOM**。

---

## 1. 基线数字（本机实测）

| 项 | 数值 | 来源 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU，driver 617.42，compute_cap 8.9 | `nvidia-smi` |
| 总显存 | 12282 MiB = **12281.5 MB** | `nvidia-smi`；`torch.cuda.get_device_properties(0).total_memory` |
| **空载已占用** | **3036 MiB**（桌面 + 浏览器等，非 WanGP） | `nvidia-smi` |
| 空载可用 | ≈ **9.2 GB** | 上两项相减 |
| torch / CUDA | 2.16.0.dev20261019+cu134 / CUDA 13.4，capability (8,9) | `torch.__version__`、`torch.cuda.get_device_capability()` |
| 20B 模型 | `ckpts/MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors` = **19.6 GB**（int8；非 pruned 版 31.7 GB） | `ls -l ckpts/` |
| 同管道子组件 | video VAE int8 **2.62 GB**、latent upscaler bf16 **0.64 GB**、veda predictor int8 **0.26 GB**、audio VAE fp32 **0.6 GB** | `models/minimax_h3/minimax_h3_main.py:26-32` |
| text encoder | Qwen3-VL-32B-Instruct layer50 int8，目录 **25 GB** | `models/minimax_h3/minimax_h3_handler.py:34-35`、`:522-526` |
| 安全系数上限 | `0.85 × 12281.5 ≈ 10439 MB`（模型可占 VRAM 上限） | `mmgp/offload.py:6231`（默认 0.85）、`:6346`、`:6545-6554`；WanGP 传入点 `wgp.py:4451` |

---

## 2. 谁在调度显存

### 2.1 唯一的调度器：一个 `mmgp.offload.offload` 实例

- 构建入口：`wgp.py:4238 def init_pipe(pipe, kwargs, profile, output_type="video", extension=None)`，实参在 `wgp.py:4574` 交给 `offload.profile(...)`，返回的 `offloadobj` 就是整条管道的显存调度器。
- 它接管的模型集合就是 `pipe` 字典：`transformer` / `vae` / `video_encoder` / `audio_vae` / `text_encoder` / `vision_encoder` / `latent_upscaler`（`models/minimax_h3/minimax_h3_handler.py:1295-1316`）。
- 它的职责：给每个模型算预算、决定谁在显存里、按 block 搬运、维护 pinned 池、选 slot、挂 forward hook（`mmgp/offload.py:4011` 起）。

**关键：`RTX_40` 不是显存 profile。** 它是**安装/依赖** profile：`setup_config.json:182-185` 定义 `RTX_20/30/40/50` → python 3.11、torch cu130、triton v37、sage v220_cu13、sparge v010_cu13、flash v210、kernels `["wgpkernels","nunchaku_cu13","gguf"]`（RTX_50 另加 light2xv），由 `setup.py:277-284` 按显卡名匹配（`if "40" in g: return "RTX_40"`，兜底也返回 `RTX_40`）。**显存 profile 只有 `profile_type` 那 6 档**，用户当前是 4。

### 2.2 profile 六档：`mmgp/__init__.py:7-29`

```python
class profile_type(int, enum.Enum):
    AutoRAM_AutoVRAM  = 0   HighRAM_HighVRAM = 1   HighRAM_LowVRAM = 2
    LowRAM_HighVRAM   = 3   LowRAM_LowVRAM   = 4   VerylowRAM_LowVRAM = 5
```
`tostr` 在 `mmgp/__init__.py:9-22`，版本 `4.4.1`（`:5`）。逐档策略在 `mmgp/offload.py:6788-6880`（docstring `:6789-6808`），WanGP 侧的下拉说明文案在 `wgp.py:12240`：

| # | 名称 | pinnedMemory | budgets（mmgp 自带） | autoPreload | 实际含义 |
| --- | --- | --- | --- | --- | --- |
| 0 | AutoRAM_AutoVRAM | False | `*`=3000，`transformer`=1200 | 否 | 预算**动态**按剩余 RAM/VRAM 调整；`pinPool=True, cpuCompute="auto"`（`:6893-6901`） |
| 1 | HighRAM_HighVRAM | **True** | `*`=3000 | **是**（`:6891-6892`） | 假设 RAM 大且显存大：尽量整模预载 |
| 2 | HighRAM_LowVRAM | **True** | `*`=3000 | 否 | RAM 大、显存小：权重全锁页，靠搬运 |
| 3 | LowRAM_HighVRAM | `"transformer"` | `*`=3000 | **是** | 只有 transformer 锁页；显存够就预载 |
| **4** | **LowRAM_LowVRAM** | `"transformer"` | `*`=3000 | 否 | **用户当前档**：transformer 锁页、逐 block 流式、不预载 |
| 5 | VerylowRAM_LowVRAM | False | `*`=3000，`transformer`=**400** | 否 | 最省：不锁页（慢 2 倍）+ 更紧的 transformer 预算 |

`_:6843-6845` 先把 `budgets["transformer"]=1200`（profile 5 改 400），`:6868-6871` 是 profile 4 的取值；`:6889` 组装最终 kwargs（`pinnedMemory / extraModelsToQuantize / budgets / asyncTransfers / quantizeTransformer / prefetchWindow`）；`:6911-6912` **overrideKwargs 是整键替换，不是合并**。

### 2.3 ★ WanGP 把预算表整个换掉了（最容易看错的地方）

`wgp.py:4253-4258`：

```python
defaults = {"transformer": 100, "text_encoder": 100, "text_encoder_2": 100,
            "*": 1000 if profile == 5 else 3000}
budgets = {name: budgets.get(name, defaults.get(name, defaults["*"])) ...}
```

后果——**mmgp 文档里的 `transformer=1200` 在这台机器上根本不生效**：

| 模型 id | mmgp profile 4 给的预算 | **WanGP 覆盖后真实生效** | 效果 |
| --- | --- | --- | --- |
| `transformer` | 1200 MB | **100 MB** | 装不下 19.6 GB → 只能逐 block 流式 |
| `transformer2`（若有） | — | 100 MB | 同上；且 profile∈(3,4) 时把 `pinnedMemory` 扩到「所有 transformer*」（`wgp.py:4264-4266`） |
| `text_encoder` / `_2` | `*`=3000 | **100 MB** | 几乎必然整模换入换出 |
| `vae` / `vision_encoder` / `audio_vae` / `latent_upscaler` / prompt-enhancer 模型 | `*`=3000 | **3000 MB**（profile 5 时 1000） | 单模可占 3 GB，与得 3GB 的邻居共存就爆 |

其它注入点：`minimumBudget = preload`（`wgp.py:4262`，用户 `*_preload_in_VRAM=0`）；`perc_reserved_mem_max`（`wgp.py:4248`，用户 0）；`smartPinning=0`（`wgp.py:4573`，因为 `smart_memory_pinning=true`）；`prefetchWindow` 仅在 `server_config` 里有 `prefetch_window` 时才传（`wgp.py:4572`，用户没有 → 保持 `None`）；profile 4.5/3.5 的关 asyncTransfers / 关 pinned 特例（`wgp.py:4268-4274`）。

### 2.4 预算怎么变成显存占位：slot + prefetch window

- 预算换算：`get_parsed_budget(b)`，`"80%"` 视为总量百分比，数字视为 MB（`mmgp/offload.py:6329-6333`）；`minimum_budget` 被 `vram_safety_coefficient × 总显存` 钳住（`:6346`）。
- block 是**按最大 tower block 尺寸预分配 slot**：`slots = [[None,None,size] for _ in range(self.model_prefetch_window(model_id)+1)]`，buffer 是 `torch.empty(size, dtype=torch.uint8, device="cuda")`（`:4361-4365`、`:4377-4378`）。
- 窗口大小：`window = planned if (planned-1)*largest <= 0.05*device_capacity else 1`（`:4229-4232`，`PREFETCH_VRAM_SHARE = 0.05` 在 `:722`）；`planned` 在 `prefetchWindow is None` 时为 **1**。
  本机：`asyncTransfers=True`、`prefetchWindow=None`（`:6848-6849`）→ **2 个 slot**；每 block ≈ 385 MB → **≈ 770 MB** 显存。若显式设 `prefetch_window=2` → 3 slot ≈ **1.15 GB**（前提 `1×385 ≤ 614 MB` ✓，会真的开 3 个）。
- 搬运走独立 stream + 单线程预取：`submit_prefetch` 用 `ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmgp_prefetch")`（`:4236`），`_load_pipelined` 注释写明「VRAM 里放 window+1 个 block，像单 block 穿梭」（`:4251`）。

### 2.5 pinned memory（锁页）

`pinnedMemory="transformer"`（profile 4）→ **19.6 GB 权重整段锁在系统内存**。这是 profile 4 相对 profile 5 的唯一实质差别（README `:17` 称预留 RAM 可让搬运「最快 2 倍」），代价是 **~19.6 GB 不可换出的物理内存**。分配顺序与上限见 `mmgp/README.md:157-158`、`:132-153` 参数表（`perc_reserved_mem_max` 用户为 0）。

### 2.6 共驻（co-tenancy）：决定「谁被踢出显存」

```python
def ensure_model_loaded(self, model_id):
    ...
    if not self.can_model_be_cotenant(model_id):
        self.unload_all(keep=[...cotenants_map 里 "\*" 的...])
    self.gpu_load(model_id)
```
（`mmgp/offload.py:5566-5577`；`can_model_be_cotenant` 在 `:4721-4728`：**要求当前所有活跃模型都在新模型的共驻名单里**）

默认名单只有互指的 text encoder 三件套：`{"text_encoder": ["vae","text_encoder_2"], "text_encoder_2": ["vae","text_encoder"]}`（`:4019-4022`），`"*"` 通配（`:4012`）。

**推论（本机重要）**：`transformer` 不在任何名单里 → 一进入 denoise，`unload_all()` 把 text encoder / vision encoder 整体请出显存；`vision_encoder` 同理，无法与任何人共处。WanGP 另给 prompt enhancer 加了 `vision_tower_model ↔ embedding_model` 互指（`shared/prompt_enhancer/loader.py:238-239`，经 `wgp.py:4333-4334` 注入）。

### 2.7 VMM allocator + spill：让「爆显存」变成「变慢」

- 选择：`VRAM_ALLOCATOR_CHOICES = ("default","vmm","vmm_spill")`（`shared/cuda_memory.py:13-19`），默认 `vmm_spill`（`:87-88`）。
- 生效判据：`mmgp_vram_allocator()` 返回 `VRAM_ALLOCATOR in ("vmm","vmm_spill")`（`:34-37`）；安装点 `vram_allocator.install(spill=allocator=="vmm_spill")`（`:132-146`）。
- 语义：`spill=True` 时 VRAM 不够就溢出到系统内存（**不报错，速度断崖**）；`spill=False` 才抛 `torch.OutOfMemoryError`（`mmgp/README.md:236-248`）。
- 副作用：`torch.cuda.memory_allocated/reserved` 被重定向（`mmgp/README.md:244`）→ **判断占用以 `nvidia-smi` 为准**，别信 torch 的 reserved。
- 本机已生效（用户 `vram_allocator=vmm_spill`），所以现象是「卡、慢、掉到几 s/it」，而不是红字 OOM。

---

## 3. 一次 20B 生成的显存时间线

| 阶段 | 谁在显存里 | 典型占用（本机 profile 4） | 证据 |
| --- | --- | --- | --- |
| ① 加载 | 什么都不该在（loading 时按 block 搬） | 基础 CUDA context，几百 MB | `mmgp/offload.py:6006`（tune_preloading）、`:4851` |
| ② 文本编码 | `text_encoder`(Qwen3-VL-32B int8) + `vision_encoder` | 预算各 100 MB → **几乎全部靠逐 block 流过**，无整模常驻 | `wgp.py:4253-4258`、`mmgp/offload.py:5901-5907` |
| ③ denoise（主力） | 只有 `transformer`：base + **2 个 385 MB slot** | **≈0.8–1.6 GB（权重侧）** + 激活 | `mmgp/offload.py:4354-4378`、`:5880-5957` |
| ④ VAE 解码 | `vae`(2.62 GB int8) + `audio_vae`(0.6 GB) + `latent_upscaler`(0.64 GB)，预算各 3000 MB | 三者**互不共驻** → 轮流进出 | `minimax_h3_handler.py:1295-1316`、`wgp.py:4253-4258` |
| ⑤ prompt 增强 / Deepy | 独立 offload 对象 `enhancer_offloadobj` | `llm_model` 预算默认 **10000 MB**（被 0.85 钳到 10439） | `wgp.py:4379`、`shared/prompt_enhancer/loader.py:237`、`:233` |

⚠️ ③ 的权重侧只占 1 GB 左右——**剩下 8 GB 全给激活和别的进程**。所以本机的墙不在权重，在激活与他人占用（见 §6）。

---

## 4. 组件 → 典型占用 / 是否可调 → 证据

| 组件 | 典型占用（本机口径） | 常驻显存？ | 可调？ | 证据（路径:行号） |
| --- | --- | --- | --- | --- |
| CUDA context / 桌面 / 浏览器 | **3036 MiB（实测空载）** | 是（非 WanGP） | ✅ 关掉浏览器/壁纸/其它 CUDA 进程 | `nvidia-smi`；`mmgp/offload.py:4027`（只用 total_memory，看不见这部分） |
| DiT 50 层（19.6 GB int8） | 每 block **≈385 MB**；常驻 2 slot（可设 3）≈ **0.77–1.15 GB** | ❌ 逐 block 流式 | ✅ `profile`、`prefetch_window`、`minimumBudget` | `models/minimax_h3/transformer.py:613`（hidden 5376 / 50 层）、`:655-658`；`mmgp/offload.py:4354-4378` |
| DiT base（非 tower：embedders/head/token_refiner） | 数百 MB 级〔推断〕 | ✅ 随模型常驻 | ⚠️ 仅随 profile `budgets` | `mmgp/offload.py:6599-6601`（`BASE_BLOCKS_OUTSIDE_TOWERS`） |
| denoise 激活 / attention | 720p·124f 估 **≈1.1 GiB / 张全序列张量**（见 §6 墙 B） | 瞬时 | ✅ 分辨率、帧数、`attention_sparsity`、滑动窗 | `models/minimax_h3/transformer.py:58-63`（patch 1,2,2）、`:805`（token 数公式） |
| text encoder（Qwen3-VL-32B int8，25 GB 目录） | 单模上限 **100 MB 预算** → 逐 block 流过 | ❌ 每次 prompt 重新搬 | ✅ 量化档（`q4_k_m`/`q2_k`）、max tokens | `minimax_h3_handler.py:34-35`、`:522-526`；`wgp.py:4253-4258` |
| text encoder 的 embed_tokens | **1.5 GB 常驻系统内存**（`lookup_on_cpu`） | ❌ 永不进显存 | ✅ 换量化档 | `models/minimax_h3/minimax_h3_main.py:244` |
| video VAE（int8，36 层 decoder） | 权重 **2.62 GB**；预算 3000 MB；解码时 **256×256 硬编码分块** | ❌ 仅 encode/decode 期间 | ⚠️ **H3 的 VAE Tiling 下拉无效**（见 §5.1） | `models/minimax_h3/video_vae.py:58-61`；`pipeline.py:522-523`；`minimax_h3_main.py:257` |
| audio VAE | **0.6 GB** | ❌ 同上 | ❌ 基本不可调 | `minimax_h3_main.py:26-32` |
| latent upscaler | **0.64 GB** | ❌ 同上 | ❌ | 同上 |
| prompt enhancer LLM | 预算 **10000 MB**（钳到 10439），`_mm_pinned_pool=True` | ❌ 独立 offload 对象 | ✅ `enhancer_enabled`、`prompt_enhancer_quantization` | `shared/prompt_enhancer/loader.py:233`、`:237`；`wgp.py:4379` |
| Deepy Prime | 与 enhancer 共用引擎；`deepy_vram_mode=unload` | ❌ 用完卸载 | ✅ `deepy_vram_mode`、`deepy_enabled` | `shared/deepy/engine.py:6602`；`plugins/configuration/plugin.py:678-684` |
| pinned 系统内存（transformer 全量） | **≈19.6 GB 物理内存** | —（非显存） | ✅ `profile 5`（不锁页，搬运慢 2×） | `mmgp/offload.py:6868-6871`；`mmgp/README.md:17` |
| spill 溢出区 | 见「墙 A」：差额溢到系统内存 | 动态 | ✅ 改 `vram_allocator=default`（改成硬 OOM 报错而非静默变慢） | `shared/cuda_memory.py:34`、`:132-146`；`mmgp/README.md:236-248` |

---

## 5. 三个专项问题

### 5.1 VAE 分块 / tiling 开关——**H3 上这个旋钮是死的**

- Wan 系（非 H3）：`models/wan/modules/vae.py:1392-1426 get_VAE_tile_size(vae_config, device_mem_capacity, ...)`：
  - 装了 MMGP 分配器时：`return 256 if vae_config == 3 or device_mem_capacity < 8000 else 512`（`:1394-1395`）；
  - 未装时按显存自动：`≥24000→cfg1`、`≥16000→cfg3`、`≥8000→cfg4`、`<8000→cfg5`（`:1400-1410`），显式预设 `use_vae_config = vae_config + 2`（`:1413`）；
  - tile 映射 `cfg1=0 / cfg2=1024 / cfg3=512 / cfg4=256 / else 128`（`:1415-1424`）。
  - 调用点 `wgp.py:7673-7695`（`device_mem_capacity = total_memory/1048576`）。
  - 同构简化版：`models/wan/modules/vae2_2.py:1342-1366`。
- UI 只有两档可选（装了分配器时）：`("Default (faster, fewer tiles)", 0)` / `("Low VRAM (slower, smaller tiles)", 3)`（`plugins/configuration/plugin.py:507-513`）；未装分配器才有 4 档（`:515-524`）。建议文案 `plugins/configuration/plugin.py:185-196`。
  - **对 Wan 系**，本机（12281 MB > 8000）两档是真有区别的：Default → tile **512**，Low VRAM → tile **256**（`models/wan/modules/vae.py:1394-1395`）。UI 的 info 文案正是写给本机这种情形：「Low VRAM forces them, for GPUs where other programs also need VRAM」（`plugins/configuration/plugin.py:512`）。
  - **对 H3，两档都无效**（见下）。
- **H3 例外（关键）**：`models/minimax_h3/video_vae.py:58-61` 虽然有 `get_VAE_tile_size`，但真正的解码入口是
  ```python
  # models/minimax_h3/pipeline.py:522-523
  def _configure_tiling(self, _tile_size):
      self.vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256)
  ```
  **形参 `_tile_size` 被完全忽略**，H3 永远 256×256 分块+重叠线性混合（`video_autoencoder.py:761-771` 的 docstring 说明 tiling 会降低显存需求并混合重叠区）。
  → **在 H3 上改 `vae_config`（VAE Tiling 下拉）对显存没有任何影响。** 想动 VAE 显存只能：改分辨率/帧数（决定 tile 个数）、或换 `lightvae`（decoder 36 层 → 12 层，`minimax_h3_main.py:257`）。
- `vae_block_size: 32`（`minimax_h3_handler.py:681`）**不是解码分块**，是分辨率对齐网格（`shared/resolutions.py:362`、`:489-490`），别搞混。
- 其它模型的同类实现：`models/ltx_video/models/autoencoders/vae.py:98`、`models/kandinsky5/kandinsky/models/vae.py:1198-1223`、`models/prism/pipeline.py:167`。

### 5.2 text encoder 是否常驻显存——**不常驻，且不与 DiT 共处**

- 预算只有 **100 MB**（`wgp.py:4253-4258`）→ 整模 25 GB 只能逐 block 流过。
- 共驻名单里没有 `transformer`（`mmgp/offload.py:4019-4022`）→ denoise 一开就把 text encoder 整体 `unload_all`（`:5575-5577`）。
- 量化档位由 profile 决定：profile 3/4/5 会把 text encoder 加进 `extraModelsToQuantize`（`mmgp/offload.py:6827-6837`、`mmgp/README.md:109`）。
- 1.5 GB 的 `embed_tokens` 被 `runtime.lookup_on_cpu` 钉在 CPU（`minimax_h3_main.py:244`），走 `hook_cpu_module`（`mmgp/offload.py:5961-5975`）——**这是故意的**，不占显存。
- 结论：text encoder 是**每轮 prompt 一次性搬运**，不是常驻；它的瓶颈是 PCIe/内存带宽，不是显存容量。

### 5.3 denoise 阶段是否逐层搬运——**是，每进一个新 block 就搬一次**

```python
# mmgp/offload.py:5901-5907
def check_load_into_GPU_needed():
    self.ensure_model_loaded(model_id)
    if blocks_name == None: ...
    elif (blocks_name != self.loaded_blocks[model_id]
          and blocks_name not in self.preloaded_blocks_per_model[model_id]
          and blocks_name != self.cpu_blocks.get(model_id)):
        self.gpu_load_blocks(model_id, blocks_name)
```
每个 block 的 `forward` 被替换成 `partial(check_load_into_GPU_needed_module, target_module)`（`:5957`），装载器挂在 `:5880` 的 hook 上。`gpu_load_blocks` 里 `pipelined = async_transfers and blocks_name != None and model_prefetch_window(model_id) > 0`（`:4909`），先 `gpu_unload_blocks` 上一个 block（`:4915-4922`）。

即：**50 层 = 每次 denoise 每 step 50 次 block 级 PCIe 搬运**（跨 step、跨 tower 的顺序会被 `tune_preloading` 学习优化，`mmgp/offload.py:6006`）。VAE 则是另一套：`_offload_hooks = ['encode','decode']`（`models/wan/modules/vae.py:935`、`vae2_2.py:815`；挂载逻辑 `mmgp/offload.py:6657-6659`），**只在 encode/decode 前后整模进出**。H3 的 `MiniMaxH3VideoVAE` 没设 `_offload_hooks`，走通用 forward hook。

---

## 6. 12GB（RTX 4080 Laptop，sm89）：三条撞墙判断

### 墙 A（最先撞）：**3 GB 已被别人占走，而 mmgp 假装有 12 GB**

- 判据：`device_mem_capacity` 取的是 **total_memory**（`mmgp/offload.py:4027`），预算上限 `0.85 × 12281 = 10439 MB`（`:6346`）；而 `nvidia-smi` 实测空载已用 **3036 MiB**，真实可支配 ≈ **9.2 GB**。
- 现象：**不会 OOM**，而是 spill 溢出（`shared/cuda_memory.py:34`、`:132-146`）→ 掉速、`s/it` 抖动、风扇狂转。
- 可调项（按性价比）：
  1. 关闭浏览器/视频/其它 CUDA 进程，把 3036 MiB 降到 <1 GB（最直接，白赚 2 GB）；
  2. `profile` 4 → **5**（`budgets["*"]` 收到 1000 MB，transformer 400 MB；`mmgp/offload.py:6873-6878`）；
  3. `enhancer_enabled=3` / `deepy_enabled=1` 常驻是额外 offload 对象（`wgp.py:4379`，`llm_model` 预算 10000 MB）→ 非必要就关，或确认 `deepy_vram_mode=unload` 已生效（`shared/deepy/engine.py:6602`）；
  4. `prefetch_window=0`（禁预取，省 385 MB；`mmgp/offload.py:4229-4232`）；
  5. 想「要错就报错」而不是静默变慢：`vram_allocator=default`（`mmgp/README.md:248`）。

### 墙 B（决定性撞墙）：**denoise 的激活，不是权重**

- 算账（〔推断〕按 Wan 系 VAE 时间压缩 4、空间 8，H3 `patch_size=(1,2,2)`，见 `models/minimax_h3/transformer.py:58-63`、`:615`）：
  - 124 帧 → latent **31 帧**；720p → latent 90×160 → patch 后 **45×80**；
  - token 数 = 31 × 45 × 80 = **≈111,600**（公式在 `transformer.py:805`：`latent_t × (latent_h/2) × (latent_w/2)`）；
  - hidden 5376、bf16 → **单张全序列激活 ≈ 111600 × 5376 × 2 B ≈ 1.14 GiB**；q/k/v 三张就是 3.4 GiB。
- 这就是为什么仓库里有 **VEDA / VSA 稀疏注意力**（`models/minimax_h3/veda/__init__.py:1`、`vsa/__init__.py`：64-token tile、per-tile top-k、稠密 text/audio 前缀；tile 128/64 行 `veda/tiling.py`、`vsa/tiling.py` 的 `BLOCK=64`、`TILE_SHAPES={64:(4,4,4),128:(4,4,8)}`），以及 `attention_mode=auto` 默认吃这套。
- 可调项：
  1. **分辨率 / 帧数**（二次方影响，最有效）：`resolution_budget` 2048×2048、`resolution_budget_tiled` 3840×2176（`minimax_h3_handler.py:688-689`），默认 `video_length=124`（`:1400`）；
  2. `attention_sparsity`（默认 **1.3**，`minimax_h3_handler.py:1411`）→ 调高更省；
  3. `sliding_window_size` 362 → 124（`:1401` / `:1352`），配合 `sliding_window_overlap=18`；
  4. `num_inference_steps` 20（影响时间不影响峰值）、`skip_steps_*` / first-block cache（`minimax_h3_handler.py:1412-1413`、`models/minimax_h3/first_block_cache.py`）；
  5. `attention_mode` 在 veda / vsa 之间换（`attention_mode=auto` 时的选择逻辑见 handler `:667` 的 `"cuda_graphs": [] if veda or fasth3 else ["transformer"]`——**veda 模式默认不开 CUDA graph**，这也是 veda 更省显存的一个侧证）。

### 墙 C（收尾撞墙）：**VAE 解码阶段的三件套轮流进出**

- 权重侧：video VAE **2.62 GB** + audio VAE **0.6 GB** + latent upscaler **0.64 GB**，预算各 **3000 MB**，且三者**互不共驻**（`mmgp/offload.py:4019-4022` 只有 text encoder 三件套互相认亲）→ 解码时轮流 `unload_all`/`gpu_load`（`:5575-5577`）。
- 激活侧：H3 **永远 256×256 分块**（`models/minimax_h3/pipeline.py:522-523`，形参被丢弃），tile 数随分辨率/帧数线性增长且**无法通过 UI 调小**。
- 可调项：
  1. 分辨率/帧数（同时减小 tile 数与每 tile 激活）；
  2. `lightvae`：decoder 36 层 → 12 层（`minimax_h3_main.py:257`）；
  3. 不要指望 `vae_config`——**H3 上它无效**（§5.1），这是本仓库当前一个真实的可调性缺口；
  4. 若必须控 VAE 峰值，唯一源码级杠杆是 `tile_sample_min_height/width`（当前写死 256）。

---

## 7. 可调项速查（本机 12GB，按性价比排序）

| 序号 | 动作 | 预期收益 | 代价 / 风险 | 证据 |
| --- | --- | --- | --- | --- |
| 1 | 关掉浏览器/其它显存占用进程 | **+2 GB 真实显存** | 无 | `nvidia-smi` 3036 MiB |
| 2 | 降分辨率/帧数（720p→512p 或 124→81 帧） | 激活与 tile 数平方级下降 | 画质/时长 | `minimax_h3_handler.py:688-689`、`:1400` |
| 3 | `attention_sparsity` 1.3 → 1.6+ | 显著降 attention 峰值 | 细节可能变糊 | `minimax_h3_handler.py:1411` |
| 4 | `sliding_window_size` 362 → 124 | 单窗 token 数降 ~3× | 长视频一致性 | `:1401`、`:1352` |
| 5 | `profile` 4 → 5 | 预算收到 1000 MB、不锁 19.6 GB 页 | 搬运慢 ~2× | `mmgp/offload.py:6873-6878`；`README.md:17` |
| 6 | `prefetch_window` = 0 | 省 385 MB slot | 失去重叠搬运 | `mmgp/offload.py:4229-4232` |
| 7 | 关 `enhancer_enabled` / `deepy_enabled` | 不再有 10000 MB 预算的第二调度器 | 失去提示词增强 | `wgp.py:4379`；`loader.py:237` |
| 8 | text encoder 换 `q4_k_m`/`q2_k` | prompt 编码阶段更快更省内存 | 语义略损 | `minimax_h3_handler.py:522-526` |
| 9 | `vram_allocator=default` | 把静默 spill 变成显式 OOM | 会直接报错 | `shared/cuda_memory.py:34`；`README.md:248` |
| 10 | ~~`vae_config`~~ | **H3 上无效**，别浪费一轮实验 | — | `models/minimax_h3/pipeline.py:522-523` |

---

## 8. 未验证 / 假设（用来指导下一轮）

1. **激活数字是推算**：§6 墙 B 的 111,600 token 依赖「时间压缩 4 / 空间压缩 8」这一 Wan 系假设；H3 VAE 的压缩率来自配置（`models/minimax_h3/components/video_autoencoder.py:706-707` 由 `math.prod(...downsample_factors)` 决定），未逐个核验。若实际时间压缩为 8，token 减半、激活减半，**墙 B 的排序不变**（仍是第一大项）。
2. **`BASE_BLOCKS_OUTSIDE_TOWERS` 的实际 base 体积未量**（`mmgp/offload.py:6599-6601`）；只知它随模型常驻。
3. **`perf` 数字全部未实测**：本文没有跑生成，`s/it`、实际峰值显存、spill 触发点都需一次真实生成用 `nvidia-smi --query-gpu=memory.used -lms 200` 采样确认。
4. `prefetch_window` 的真实取值建议用 `verboseLevel=2` 打一遍 profile 参数（`mmgp/README.md:111-116`、`mmgp/offload.py:6903-6909` 会把每个键的值与 override 打印出来），这是唯一零成本确认预算的办法。
5. `enhancer_enabled=3` 的确切语义（三种增强器变体）未在本文定论；已确认的是它必然加载一个 `llm_model` 进 pipe（`wgp.py:4326-4334`）或另建 `enhancer_offloadobj`（`:4379`）。
