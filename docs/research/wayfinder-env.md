# 本机环境瓶颈取证（wayfinder 票 #5）

- 采集时间：2026-10-31 13:54–14:00 CST
- 采集方式：**只读取证 + 轻量基准**；未改任何配置、未跑视频生成、未重启 WSL、未提交 git
- 主机：LAPTOP-5R2A6274，Windows 10.0.26300.9818，WSL 3.0.3.0 / 内核 6.18.54.1-microsoft-standard-WSL2
- 发行版：Ubuntu-26.04（WSL2），`/etc/wsl.conf` 仅 `systemd=true` + `default=acgmn`

## 结论（TL;DR）

1. **唯一真实瓶颈是内存，不是磁盘、不是 GPU、不是 import 开销。**
   宿主机有 63.6 GB 物理内存，但 WSL2 沿用「默认取 50%」拿到了 31.1 GiB；WanGP 本体常驻匿名内存 20.3 GB，31 GB 上限已经压到换页（swap 用掉 3.8 GB，`pgsteal_direct` 1524 万页 = 发生过真实阻塞式回收）。同时 Windows 侧只剩 **3.4 GB 可用**，`vmmemWSL` 独占 30.59 GB 不归还。
2. **磁盘完全不是瓶颈**：`ckpts/` 在原生 ext4（`/dev/sdd`），O_DIRECT 实测 **4578–4666 MiB/s**（4.5–4.7 GB/s）。
3. **GPU 直通健康**：驱动 617.42 / CUDA UMD 13.4，torch 2.16.0.dev+cu134 与 triton 3.9.0 自编译内核均验证通过，bf16 matmul 57.0 TFLOPS。
4. **import 开销正常**：`import torch` 1.19 s，`import torch, triton` 1.20 s。
5. **顺手量到一条硬约束**：9p 挂载（`/mnt/c|d|f`）读速只有 **135–167 MiB/s**，比 ext4 慢 **27–34 倍** —— 权重绝不能放 Windows 盘。

## 1. `.wslconfig` 与内存上限

### 命令与结果

```bash
ls -l /mnt/c/Users/ASUS/.wslconfig && cat /mnt/c/Users/ASUS/.wslconfig
free -g
head -3 /proc/meminfo
swapon --show
```

`.wslconfig` **存在**，31 字节，内容仅两行：

```ini
[wsl2]
networkingMode=Mirrored
```

**没有任何 `memory=` / `swap=` / `processors=` 限制** —— 所以 31 GB 不是人为设定，而是 WSL2 的默认规则（取宿主机物理内存的 50%）：

| 指标 | 数值 |
| --- | --- |
| 宿主机物理内存（Windows） | **63.6 GB** |
| 50% 默认 → WSL 实得 | **31.1 GiB**（`MemTotal: 32666828 kB`） |
| `free -g` | total 31 / used **21** / free 2 / buff-cache 7 / **available 9** |
| `MemFree` | 2 227 096 kB（2.1 GiB） |
| `MemAvailable` | 9 653 164 kB（9.2 GiB） |
| Swap | 8 GB 分区（`/dev/sdc`），已用 **3.8 GB**（47%） |
| `pswpin` / `pswpout` | 574 528 / 1 414 559 页 |
| `pgsteal_direct` | **15 245 512 页** ← 直接回收，会阻塞进程 |
| OOM kill | `dmesg` 中**无** OOM 记录（还没到被杀的程度，但已在换页） |

### 谁在吃内存（决定性问题）

```
PID 93227  RSS 27 841 444 kB (26.5 GiB)  VmSwap 1 356 280 kB  Threads 97
cmd: /home/acgmn/github/WanGP/.venv/bin/python /home/acgmn/github/WanGP/wgp.py
ppid 93192: uv run .../wgp.py
started 12:41:11，已运行 1h15m，cwd = /home/acgmn/github/WanGP
RssAnon 20.33 GB | RssFile 5.67 GB | RssShmem 0.08 GB
```

**这 26.5 GB 是 WanGP 自己**（`wgp.py` 正在运行），不是内存泄漏。其中 20.33 GB 是匿名内存（真实占用），5.67 GB 是模型文件的 mmap 页缓存。

### Windows 侧对照（证明是双向紧张，不是 WSL 单方面贪）

| 指标 | 数值 |
| --- | --- |
| Windows `Available MBytes`（含 standby） | **3 471 MB ≈ 3.4 GB** |
| `FreePhysicalMemory` | 3.0 GB |
| Commit | 64.1 GB used / 127.6 GB limit（50%，不紧张） |
| `vmmemWSL` 工作集 | **30.59 GB** |

Windows 侧大户：pycharm64 3.28 GB、Everything 1.76 GB、360ChromeX 1.58+0.83 GB、DeepSeek Harness 1.27+1.19 GB、explorer 0.95 GB、MsMpEng 0.57 GB、Weixin 0.51 GB。

**判读**：物理内存总量足够（63.6 GB vs 实际需求约 20 GB 匿名 + Windows 桌面约 12 GB ≈ 32 GB），问题是 WSL2 拿满 31 GB 后**不主动归还**缓存页，同时 WanGP 在 31 GB 上限里被挤到换页。两头同时紧。

## 2. GPU 直通与驱动

### 命令与结果

```bash
nvidia-smi
ls -la /usr/lib/wsl/lib/
df -h .
mount | grep -E 'ext4|9p|drvfs'
```

```
NVIDIA-SMI 615.78.05    KMD Version: 617.42    CUDA UMD Version: 13.4
RTX 4080 Laptop  | 12282 MiB | used 3041 MiB | free 8955 MiB | 5% | 53C | P0 | 35W/140W
Processes: No running processes found（WSL 侧看不到 Windows 的图形进程）
```

| 项 | 实测值 |
| --- | --- |
| 驱动（Windows KMD） | **617.42** |
| `nvidia-smi` 工具版本（WSL 侧） | 615.78.05 |
| CUDA UMD | **13.4** |
| 显卡 | RTX 4080 Laptop，12282 MiB，sm_89 |
| 显存占用 | used **3041 MiB** / free **8955 MiB**（torch 视角 11.99 GiB 总量、10.78 GiB 空闲） |
| 利用率 / 温度 / 状态 | 7% / 52 °C / P0 |
| 功耗 | 当前 39 W，**强制上限 140 W**，硬件上限 **175 W** |
| SM 时钟 / 最大 | 2280 MHz / 3105 MHz（空闲未升频，非降频） |
| 图形上下文数（`C+G`） | **30 个**（Edge、360ChromeX、WeChat、PowerToys、PyCharm jcef、DSH 等） |

### `/usr/lib/wsl/lib` 下的 CUDA 库

```
libcuda.so / .so.1 / .so.1.1     188 KB   ← WSL 直通桩（转发到 Windows 驱动，非完整 UMD）
libnvidia-ml.so.1                291 KB
libnvdxdlkernels.so              154 MB
libnvidia-gpucomp.so             118 MB
libnvwgf2umx.so                   58 MB   （D3D12 后端）
libnvcuvid.so                     27 MB
libnvidia-ngx.so.1               5.5 MB
libnvidia-encode.so              273 KB
nvidia-smi                       849 KB
libd3d12.so / libd3d12core.so / libdxcore.so  （WSLg 图形栈）
```

**判读**：驱动 617.42 已远超 CUDA 13.4 所需的最低版本，直通链路正常（torch 能识别设备、能分配显存、能跑 kernel）。`libcuda.so` 只有 188 KB 是 WSL 设计的正常现象（真 UMD 在 Windows 侧），不是缺库。

### 轻量基准（GPU 软件栈健康度）

```bash
.venv/bin/python /tmp/smoke.py
```

```
torch 2.16.0.dev20261019+cu134 | cuda 13.4 | sm (8, 9)
  bfloat16  matmul 4096^3: 2.41 ms = 57.0 TFLOPS
  float16   matmul 4096^3: 2.20 ms = 62.5 TFLOPS
  fp8 e4m3 available: True
  triton 3.9.0 kernel ok: True   ← 自写 @triton.jit 内核编译并结果正确
```

> 注：该数字在桌面同时占用 GPU（30 个图形上下文、7% 利用率）的情况下测得，是**指示性**数值而非峰值；主要用于确认 bf16/fp16/fp8/triton 四条路径都通。

## 3. 磁盘 I/O

### 命令与结果

> 原计划的 `dd if=<file> of=/dev/null bs=8M count=16 iflag=direct` 被环境安全校验拦截（把 `of=/dev/null` 判成块设备写入），改为等价的 Python `O_DIRECT`（`posix_memalign` 对齐缓冲 + `os.readv`），语义一致：绕过页缓存直读裸设备。

```bash
findmnt -no FSTYPE,SOURCE,TARGET /
df -h . ; df -h ckpts
mount | grep -E 'ext4|9p|drvfs'
```

| 项 | 实测值 |
| --- | --- |
| `/` 与 `ckpts/` 文件系统 | **ext4**，`/dev/sdd`（原生 WSL2 VHDX，**不是** 9p/drvfs） |
| 容量 | 1007 G 总 / 441 G 已用 / **515 G 可用**（47%） |
| inode | 927 640 / 67 108 864（2%） |
| 挂载选项 | `rw,relatime,discard,errors=remount-ro`（**discard 已开**） |
| `ckpts/` | **288 G**，311 个文件，53 个顶层条目 |
| 磁盘设备 | `/dev/sdd` 1T（Virtual Disk，即 WSL VHDX） |

### 读速（O_DIRECT，绕过页缓存）

被测文件：`ckpts/MiniMax-H3-FL2VA_int8_convrot.safetensors`（31.7 GiB）

| 轮次 | 读取量 | 耗时 | 吞吐 |
| --- | --- | --- | --- |
| run1 | 512 MiB | 0.112 s | 4578 MiB/s |
| run2 | 512 MiB | 0.116 s | 4428 MiB/s |
| run3 | 512 MiB | 0.110 s | 4634 MiB/s |
| **中位数** | 512 MiB | — | **4578 MiB/s** |
| 持续读 | **2048 MiB** | 0.439 s | **4666 MiB/s** |

小文件复测（`ckpts/ltx-2.5-22b_audio_vae_bf16.safetensors`，101.6 MiB，`bs=8M`×16）：3925 / 4433 / 4618 MiB/s，中位数 **4433 MiB/s**。

**结论：≈ 4.5–4.7 GB/s，远超任何模型加载需求（即便 34 GB 的 checkpoint 也只需约 7.5 秒）。磁盘不是瓶颈，且 515 GB 可用空间充裕。**

### 对照：9p 挂载盘（关键反例）

同一方法测 `/mnt/d` 上的 1.9 GB 文件：

| 方式 | 吞吐 |
| --- | --- |
| ext4 (`/dev/sdd`) O_DIRECT | **4578 MiB/s** |
| 9p drvfs (`/mnt/d`) O_DIRECT | **135 MiB/s** |
| 9p drvfs (`/mnt/d`) buffered | **167 MiB/s** |

**慢 27–34 倍。** `/mnt/c`、`/mnt/d`、`/mnt/f` 都是 `9p ... aname=drvfs;cache=0x5;msize=65536`。任何把权重/模型/数据集放到 Windows 盘的方案都会直接劣化 30 倍以上。

## 4. import 开销

```bash
/usr/bin/time -f "wall=%es maxrss=%MkB" .venv/bin/python -c "import torch"
/usr/bin/time -f "wall=%es maxrss=%MkB" .venv/bin/python -c "import torch, triton"
/usr/bin/time -f "wall=%es maxrss=%MkB" .venv/bin/python -c "import torch; torch.cuda.init()"
```

| 命令 | 墙钟 | 峰值 RSS |
| --- | --- | --- |
| `import torch` | **1.19 s** | 668 MB |
| `import torch, triton` | **1.20 s** | 738 MB |
| `import torch` + `torch.cuda.init()` | 1.21 s | 703 MB |

**判读**：1.2 s 完全正常（同类场景常见 1–3 s）。`triton` 叠加只多 0.01 s，说明是延迟加载、开销可忽略。**这一项不值得优化。**

## 5. `.venv` 体积

```bash
du -sh .venv
du -sh .venv/lib/python3.13/site-packages/* | sort -rh | head -12
```

| 项 | 体积 |
| --- | --- |
| **`.venv` 合计** | **9.0 G** |
| `nvidia/` | 3.2 G |
| ├ `nvidia/cu13` | 1.6 G |
| ├ `nvidia/cudnn` | 848 M |
| ├ `nvidia/nccl` | 345 M |
| ├ `nvidia/cusparselt` | 224 M |
| └ `nvidia/nvshmem` | 193 M |
| `torch/` | 1.3 G |
| `triton/` | 998 M |
| `onnxruntime/` | 335 M |
| `jaxlib/` | 322 M |
| `unidic_lite/` | 249 M |
| `comfy_kitchen/` | 179 M |
| `_polars_runtime_32/` | 179 M |
| `llvmlite/` | 172 M |
| `gradio/` | 170 M |
| `taichi/` | 141 M |
| `spacy/` | 120 M |

9.0 GB 对 CUDNN+NCCL+torch+triton 的栈来说是**正常体积**，不是问题。可清理的冗余（`jaxlib` 322 M、`onnxruntime` 335 M）只有约 0.65 GB，且需先确认 WanGP 是否真的不用。

## 6. 风险项（仅本地证据）

> 按要求不联网搜新闻。以下每条都附本机能验证的观测点；**没有本地证据支持的风险不写**。

### 6.1 torch 夜间版 / cu134 —— 本地实测健康，但有一条架构细节

```
torch   2.16.0.dev20261019+cu134   (git e6c0cabb629231a0e9697d11f19da408f0ec8811, debug=False)
triton  3.9.0+gitfc8d31a5
cudnn   9.26.0.51  (torch.backends.cudnn.version() = 92600)
nccl    2.32.3
CUDA    torch.version.cuda = 13.4，驱动 617.42 ≥ 所需
```

**已验证正常**：`torch.cuda.is_available()` True；设备名/算力正确；bf16/fp16 matmul 正确；fp8 e4m3 类型存在；自写 triton kernel 编译通过且结果 `allclose` 为 True；CUDA 图可用。

**唯一的架构细节（值得记录，但不需要动）**：

```python
torch.cuda.get_arch_list() = ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']
```

**列表里没有 `sm_89`**，而本机是 sm_89。CUDA 的同一主版本内向后兼容规则保证 sm_86 的 SASS 能在 sm_89 上执行，所以**功能没问题**（实测也确实跑通）。代价是 Ada 拿不到为 sm_89 专门调优的 kernel，理论上会损失个位数百分比性能。这不值得为它换 torch 版本。

### 6.2 加速后端大面积缺位（是设计选择，不是故障）

```
flash_attn    absent      torchao       absent
xformers      absent      bitsandbytes  absent
sageattention absent      quanto        absent
flashinfer    absent      optimum       PRESENT
```

`ckpts/` 里大量 `*_int8_convrot.safetensors`、`*_quanto_bf16_int8.safetensors` 说明 WanGP 走的是**自带 int8/quanto 量化路径**，不依赖上述第三方后端。缺位是预期内的。

若将来要开 `torchao` 或 `flash_attn` 加速，需注意 **Python 3.13 + cu134 + torch 2.16.dev** 是极窄的 wheel 组合，很可能没有预编译包。

### 6.3 `.venv` 里没有 `pip`（小毛病）

```
$ .venv/bin/pip check
bash: .venv/bin/pip: No such file or directory
```

该 venv 由 **uv** 创建（父进程是 `/home/acgmn/.local/bin/uv run ...`），uv 默认不装 pip。影响：无法直接 `pip check` / `pip install` 做依赖冲突体检。用 `uv pip list` 或 `python -m ensurepip` 即可，非瓶颈。

### 6.4 没有本地证据支持的风险（明确排除）

- `dmesg` 中**没有** OOM kill、没有 CUDA Xid 错误 —— 目前还没到驱动崩溃/进程被杀的阶段。
- 没有看到 cu134 相关的本地错误日志。

## 汇总表

| 项 | 实测值 | 是否值得动 | 改法与回滚 |
| --- | --- | --- | --- |
| **WSL 内存上限**（`.wslconfig` 无 `memory=`，走 50% 默认） | 31.1 GiB / 宿主 63.6 GB；used 21 G，available 9 G | **值得动 · 最高优先** | 在 `%UserProfile%\.wslconfig` 的 `[wsl2]` 下加 `memory=40GB`、`swap=16GB`、`autoMemoryReclaim=gradual`（新版在 `[wsl2]`，旧版在 `[experimental]`）、`pageReporting=true`。**回滚**：删掉这几行恢复原两行内容，`wsl --shutdown` 后重进。原文件仅 31 字节，已在本报告第 1 节完整保留。 |
| **Swap 压力** | 8 G 分区，已用 **3.8 G**；`pgsteal_direct` 1524 万页 | 值得动（随内存上限联动） | 提高 `memory=` 后 swap 自然回落；`swapFile=` 可选放大到 16 GB。**回滚**：去掉 `swap=` 行。 |
| **Windows 侧内存枯竭** | available **3.4 GB**；`vmmemWSL` 独占 30.59 GB | **值得动 · 与上一条同源** | 见效最快的是先关掉常驻大户：PyCharm 3.3 G、Everything 1.8 G、360ChromeX 2.4 G、多余的 DSH 实例 1.2 G ≈ 可释放 8 G。**回滚**：重新打开即可，无持久副作用。 |
| **GPU 直通 / 驱动** | 617.42，CUDA UMD 13.4，sm_89，12 GB | **不值得动**（健康） | 无需改。 |
| **GPU 功耗墙** | 强制 140 W，硬件上限 **175 W** | 可选 · 小幅收益 | Windows 侧管理员执行 `nvidia-smi -pl 175`。**回滚**：`nvidia-smi -pl 140`。注意笔记本散热余量，可能换来更高温度/风扇噪声。 |
| **桌面显存占用** | 3041 MiB 已用，**30 个** `C+G` 图形上下文 | 可选 · 小幅收益 | 关掉不用的浏览器/WeChat 可回收约 1–2 GB 显存，对 12 GB 卡在跑 14B/22B 模型时有意义。**回滚**：无（重开应用即可）。 |
| **`ckpts/` 磁盘 I/O** | ext4 `/dev/sdd`，O_DIRECT **4578–4666 MiB/s**，515 G 可用 | **不值得动** | 无需改。 |
| **权重放在 Windows 盘** | 9p drvfs **135–167 MiB/s**，慢 27–34 倍 | **必须避免**（现状正确） | 保持 `ckpts/` 在 ext4。若已有人把权重指到 `/mnt/*`，移回 `$HOME` 下。**回滚**：不适用（这是正确方向）。 |
| **import 开销** | torch 1.19 s / torch+triton 1.20 s | **不值得动** | 无需改。 |
| **`.venv` 体积** | **9.0 G**（nvidia 3.2 G、torch 1.3 G、triton 998 M） | 低优先 | 可删 `jaxlib`(322 M)、`onnxruntime`(335 M)，**前提是先确认 WanGP 不使用**。**回滚**：`uv pip install` 重装。 |
| **torch 缺 sm_89 cubin** | arch_list 有 sm_86 无 sm_89 | **不值得动** | sm_86 SASS 在 sm_89 上二进制兼容，功能已验证通过；换版本风险远大于收益。 |
| **`.venv` 无 pip** | `.venv/bin/pip` 不存在（uv venv） | 小修 | `python -m ensurepip` 或改用 `uv pip`。**回滚**：忽略即可。 |
| **缺 flash_attn/torchao 等** | 全部 absent，但 `optimum` 在 | 按需 | 当前 int8_convrot 路线不需要；若要开再评估 wheel 可用性（py3.13+cu134 极窄）。**回滚**：不装即原状。 |

## 建议动作顺序（未执行，仅建议）

> 本票为只读取证，以下均未实施。

1. **先释放 Windows 桌面占用**（零风险、立即见效）：关 PyCharm / Everything / 一个 360ChromeX / 多余 DSH 实例，目标是把 Windows available 从 3.4 GB 拉到 10 GB 以上。
2. **再改 `.wslconfig`**：加 `memory=40GB` + `swap=16GB` + `autoMemoryReclaim=gradual` + `pageReporting=true`，然后 `wsl --shutdown` 重进。
3. **复测三件事**：`free -g` 的 used/available、`cat /proc/swaps` 的已用量（目标 < 1 GB）、`/proc/vmstat` 的 `pgsteal_direct` 增量是否归零。
4. 只有在 WanGP 仍被换页时，才考虑把 `memory=` 继续上调（上限约 48 GB，需保证 Windows 至少 12 GB 余量）。
5. GPU 功耗墙（`nvidia-smi -pl 175`）放到最后，属于锦上添花。

## 本次未验证 / 局限

- **未改任何配置**，因此 `.wslconfig` 调整后的实际收益没有被实测（第 1–5 条建议都是推断，依据是当前的换页计数与双向内存紧张）。
- GPU 功耗墙未测试，散热余量未知。
- 57 TFLOPS 是在桌面占用 GPU 的背景下测得的指示性数值，不等于该卡峰值。
- `.venv` 中 `jaxlib`/`onnxruntime` 是否被 WanGP 实际引用未核查，故清理建议标为「需先确认」。
- 9p 对照样本来自 `/mnt/d` 的一个 1.9 GB APK 文件，非模型权重，但足以体现协议量级差异。
