# DeepInfer 导读：一个把「生产集群」当第一目标的推理框架（2026-10-20）

读者：本项目作者（正在研究小批量 decode 执行开销、做过 min-p 内核对照与 HF3FS 页淘汰补丁）。
目的：读一遍就知道这仓库该看哪几个文件、每个设计在解决什么、哪些数字能引用、哪些不能拿来对标本机。

资料来源（全部为公开材料，本地解包在 `.tmpfetch/DeepInfer-main`，本轮结束已删）：

- `README.md`：定位、支持模型、依赖、未开源清单。
- `doc/inference_deployment_qa.md`（312 行）：**信息量最大的一份**，几乎是 DeepSeek 把内部集群设计讲了一遍。
- `examples/README.md`：benchmark 口径、环境变量、典型命令。
- 代码本身：`deepinfer/`、`csrc/engine/`、`deepinfer/model_executor/layers/tilelang_ops/`。

---

## 1. 一句话定位

DeepInfer 是 DeepSeek 大规模在线服务的推理框架，本次开源的是**离线推理版**：能跑通模型、能复现他们的 kernel/调度/投机解码/结构化输出，但**不含**完整服务所需的 KV cache 系统、E/P/D 三组件的进程级分离、以及 prefill/decode 副本间的负载均衡。

血统写得很直白：**最初主要基于 vLLM**，参考了 SGLang 与 TensorRT-LLM。所以读它的正确姿势不是"学一个全新架构"，而是"看一个 vLLM 分支为了生产集群改了什么"。凡是和 vLLM 不一样的地方，就是他们踩过的坑。

只支持一个模型：`deepseek-ai/DeepSeek-V4.1-Flash-1017`。这不是"通用框架"，是"我们线上那个模型的服务栈"。硬件两条腿：NVIDIA SM100 和华为昇腾 950（`DEEPINFER_ACCELERATOR=nvidia|ascend`）。

许可证 Apache-2.0。

---

## 2. 为什么值得读：它把「部署形态」当成了设计输入

大多数开源推理框架的默认形态是"一台 8 卡机 + TP"。DeepInfer 的部署单元是：

```
Entrypoint（路由、前缀匹配）
   ├── VE  : 视觉编码器，单卡跑图像
   ├── PE  : prefill 引擎，普通 8 卡机，EP2–8 / DP2–8
   ├── CE  : decode 引擎，64–128 卡超节点，EP64–128 / DP64–128
   └── 专用存储集群（3FS）：放 paged block KV，约 100 PiB SSD
```

三个硬指标把整个架构钉死了（都在 `doc/inference_deployment_qa.md`）：

| 目标 | 数字 | 架构后果 |
| --- | --- | --- |
| 规模 | 单集群 **155 万亿 token/天** | 调度、路由、存储都必须是分布式的，不能靠单点 |
| 成本 | 单 GPU 收入 **38 美元/GPU·小时** | 决定"哪些机器必须打满、哪些可以浪费" |
| KV 保留 | **> 72 小时** | 必须外置到独立存储集群，HBM/主机内存都装不下 |

最后一个数字是理解全篇的钥匙。72 小时 × 155 万亿 token/天的前缀命中需求，意味着 KV 总量远超任何一个 GPU 集群的内存；而只要 KV 存在远端，**"命中"就必须在网络里发生**，于是路由、负载均衡、重算策略三者互相牵制。DeepInfer 大部分"奇怪"的设计都是这条链的产物。

---

## 3. 四个技术支柱

下面每个都按"问题 → 做法 → 代价/数字"写。数字全部来自上面的公开材料，出处随句标注。

### 3.1 Decode 用 Wide-EP，不用 TP

**问题**：decode 是访存密集型。一步 decode 每个 rank 都要把 MoE 的 expert 权重读一遍（HBM → SM），读的量决定了那一步的下限时间。TP 把权重切小，但每层都要 all-reduce 激活，通信随层数累加；EP 把**专家**切开，每个 rank 只读自己那几个专家。

**做法**：decode 用 EP64–128（"Wide-EP"），attention 用 DP（`#DP = #EP`）。prefill 反过来，EP2–8 就够（算力密集，切太碎通信占比高）。QA 文档的原话是 prefill"在 HBM 允许下取最小 EP"，decode"取最大"。

**代价**：EP 越大，一次 MoE 前向要 all-to-all 的 rank 越多。所以必须有 DeepEP 这样的通信库，以及 EPLB 来平衡各专家的负载。文档里给的对比是 EP8 vs EP64 的 TPS 曲线——Wide-EP 明显更高，但前提是 EP 组在一个 scale-up domain 里（超节点内 NVLink/高速互联）。

**本项目可借鉴的判据**：这条思路和"小批量 decode 步固定开销"是同一件事的两面。单卡 4.8 ms 的 decode 步里，能省的绝对时间有上限；集群里的做法是**把每步要读的字节减少**（EP），而不是把每步的 kernel 变快。读这个仓库时要一直问："这个优化减少的是字节、延迟，还是同步次数？"

### 3.2 EPLB：冗余专家 + 近期统计 + 周期性重排

**问题**：专家负载天然不均。热门专家所在的卡成为瓶颈，整步等它。

**做法**（`deepinfer/eplb/`，`__init__.py` 19 KB + `policy.py` + `shuffle.py`）：
1. 每张卡多放几个"冗余"专家（物理副本）。
2. 统计最近一段窗口内每个专家的实际负载。
3. 周期性重排：把冗余副本挪到热点卡上。权重通过 NVLink 的 `symm_mem` 洗牌搬运，不落盘。

**数字**（QA 文档默认值）：

| 阶段 | 冗余专家/卡 | 统计窗口 | 重排周期 | 均衡度 `moe_lb` |
| --- | --- | --- | --- | --- |
| Prefill | 4 | 100 步 | 每 1000 步 | 1.074 (EP4DP4) / 1.045 (EP2DP2) |
| Decode | 1（EP64 取整后每卡 7 个，不含 shared expert） | 1000 步 | 每 10000 步 | 1.090 |

`moe_lb` 是他们自己的均衡指标（1.0 = 完美），1.09 意味着最慢专家比平均慢 9%。**这是"能公开的数字"的典型样子**：不是"提升了 X%"，而是一个可对照的物理量。

工程细节（env 里能看到）：`DEEPINFER_MOE_LB_STAT_INTERVAL`、`DEEPINFER_EXPERT_COUNT_WINDOW_SIZE`、`DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL`、`DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS`（默认 32，且只作下界，按 ep_size 上取整）。

**坑（文档里明写）**：不同窗口的统计会相互干扰，benchmark 里要**剔除与 MoE LB 统计 flush 冲突的区间**——也就是说，即使你不改代码，统计系统本身会周期性拖慢一步。这和我们测 min-p 时"测量装置会改变被测对象"是同一类问题。

### 3.3 分层 KV：谁会命中，决定了 KV 放哪

这是全篇最值得抄的一节。KV 分三层，每层放的东西不同：

| 位置 | 放什么 | 为什么 |
| --- | --- | --- |
| GPU HBM | 运行中请求的 block KV + state KV | 只有正在跑的需要最快 |
| 本地主机内存 | SWA KV（单副本） | 窗口小、按 128 token 定长、跨机命中走 400G VLAN 就够 |
| 专用 3FS 集群 | paged block KV（无应用级副本） | 让**任意** prefill 机都能命中**任意**前缀 |

关键决策：**不用分层存储，用专用存储集群**。理由三条，都很硬：
1. 如果 KV 只在"有亲和性的"机器上，路由就被前缀绑定，负载均衡没法做。
2. 存储的扩缩容与算力解耦（存储贵、算力更贵）。
3. 计算节点故障率比存储节点高一个数量级以上，KV 放在计算节点上等于把可用性交给最不可靠的一环。

3FS 参数：**10+2 EC**（读时打 11 个节点，任意 10 个响应即可）；约 **100 PiB** SSD；约 **14.4 TB/GPU** 逻辑容量；TTL：agent 流量 **3 天**、chatbot **7 天**。他们算过：3 天 TTL 对 chatbot 命中率只损失 **0.5%–1%**——这就是"保留 72 小时"这个目标的量化来源。

访问开销怎么藏：
- **前缀匹配藏不掉**：Entrypoint 要在**路由之前**知道命中多长，才能决定发给谁。所以只能"只查元数据"（Entrypoint 对本地 kvcache-store 做 hash-chain 匹配，CHECK 角色）。
- **取 KV 藏得掉**：独立进程异步搬到主机共享内存。
- **swap-in 用 GDR**（RDMA 网卡直接打 PCIe），不用 CUDA copy engine——因为细粒度 block 拷贝下 copy engine 表现很差。这条是纯经验，文档直接写了"实测差"。

### 3.4 SWA 与 bounded replay：把"重算"变成常数

V4.1 用了滑动窗口注意力（SWA）+ YOCO 结构。窗口注意力的问题在于：**前缀缓存命中时，最后一段的 SWA 状态是缺的**（窗口内每个位置都要看到前 128 个 key 的累积状态，而 block KV 按 128 对齐存）。

**做法**：
- SWA KV 存在 state cache 的定长 `StateBlock` 里，槽内分成 `[stored region: SWA][dropped region: YOCO consumer SWA]`，V4.1 只存前 20 层。
- 只存 128 token 整倍数位置。每序列磁盘占用 = 20 层 × 134 token × 528 B = **1,415,040 B**（528 B/token 是 MXFP8 SWA KV 格式）。
- 命中判定：先查 block KV 得 L1（128 的倍数），再查 SWA KV：`[L1−128, L1]` 有就命中 L1，否则退到 L1−128。最大命中长度 ≤ L−134（128 窗口 + 6 个回溯位）。
- **bounded replay**：命中后只重算最近 128 token，并且把注意力窗口**截断在重放段内**（位置 i 只对 `max(s, i−127)..i` 的 key 算）。于是重算代价与上下文长度、层数**无关**。

两处用它：
1. Encoder 侧重建 SWA KV（block KV 原样复用，不重算不覆盖）。
2. Decoder 侧（CED 结构下 decoder block KV 由最后一层 encoder hidden 投影而来）：精确重建要跑最后 2560 token × 20 层，改成只重算最后 128 token、结果仅供 decode 不进 prefix cache，**近乎把 prefill 总算量减半**。

**这是本轮读到的最漂亮的一个"用结构换常数"**：不是把重算变快，而是证明"重算的那部分对结果没有贡献"，然后把它删掉。同时它解释了一个容易被忽略的接口问题——**命中长度由 block KV 决定、最后一块是否重算由 SWA KV 决定**，两者是两个数据源的联合判定。

### 3.5 DSpark：不等全部 draft，按置信度决定验证几个

**问题**：投机解码默认一次验证全部 N 个 draft token。但越往后的 draft token 越容易被拒，验证它们是纯浪费。

**做法**（`deepinfer/model_executor/models/mtp/dflash.py`，env 里 `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE`）：draft 模型一次出 5 个 token，**Confidence Head** 预测每个 draft token 的接受率；按期望接受率排序，选一个使**有效吞吐最大**的前缀子集去验证（变长验证）。

**数字**（QA 文档，生产环境）：

| 并发/rank | 16 | 32 | 48 | 60 | 72 | 96 |
| --- | --- | --- | --- | --- | --- | --- |
| 动态验证长度 | 4.44 | 3.65 | 3.94 | 3.58 | 3.24 | 3.24 |
| 吞吐增益 | −1.5% | +1.6% | +3.6% | +5.4% | +6.7% | +11.0% |

平均每步验证 draft 数 5.00 → **3.2–4.4（约 −30%）**，而平均接受长度几乎不变（3.29 vs 3.20），EP64/DP64 下整体 **+4–7%**（~60 并发/rank）。

**注意低并发是负收益**（16 并发 −1.5%）：低并发时 GPU 有富余，少验证几个 draft 省不下时间，反而少了可能的接受。这就是为什么它是"confidence-scheduled"而不是"固定截断"。

### 3.6 异步调度：把 CPU 该干的活塞进 GPU 在跑的时间

**问题**：decode 原生步序是 `schedule(N) → forward(N) → sample(N) → post-process(N)`。post-process 要读采样结果（GPU→CPU 同步），同步之后 GPU 空转，CPU 才去 schedule(N+1)。小 batch 时这一步空转占比很大。

**做法**（`csrc/engine/zos.cpp/.hpp`，env `DEEPINFER_ZERO_OVERHEAD_SCHEDULING`；参考 SGLang 与 NanoFlow）：

```
GPU:  forward(N) → sample(N) → forward(N+1) → sample(N+1)
CPU:  schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)
```

CPU 永远在处理**上一步**的结果，GPU 永远不等 CPU。在 `deepinfer/engine/llm_engine.py` 里能看到这条流水线的形状：`step()`、`_execute_model_async_step()`、`post_process(step_state)`、`pre_sampler_sync`。

**它和 DSpark 撞车**：schedule(N+1) 的时候，第 N 步的 confidence 还没回到 CPU。解法很干脆——**CPU 只定总预算，逐请求验证长度由 GPU 上的 TileLang kernel 重算**（`layers/tilelang_ops/schedule_mtp_varlen.py`：累乘、全局排序取 top-k、融合 remap scatter，写回 pinned buffer）。

**这条最值得本项目借鉴**：它不是"藏 50 µs 计算"，而是**删除一次同步**。同步点一旦存在，代价是"CPU 与 GPU 的串行段之和"，而不是"CPU 那段时间"。

### 3.7 另外两块

- **结构化输出**（`deepinfer/structure_output/`：`compiler.py`、`matcher.py`、`mask.py`、`so_matcher_builder.py`）：格式编译与推理解耦，SO 在专用机器上异步编译成语法自动机；推理时每步推进自动机、求合法 token 集、生成 mask 施加到采样。**编译在推理路径外**是关键——否则第一个请求要为编译付费。
- **batch-invariant 推理 + 确定性采样**（`DEEPINFER_BATCH_INVARIANT=1`，仅 NVIDIA）：同一个请求，不管 batch 里有多少其他请求，输出逐 token 相同。这是**可复现性**需求（也是我们做 A/B 时最缺的东西）：batch 变化导致 kernel 归约顺序变化 → logits 变化 → 采样漂移。DeepInfer 直接把这个自由度去掉，代价是放弃了部分 batch 相关的 kernel 优化。

---

## 4. 代码地图：从哪进去

```
DeepInfer/
├── csrc/                       C++/CUDA：engine（调度器本体）、graph_timer、input_metadata
│   └── engine/
│       ├── scheduler.hpp       struct Scheduler / SchedulerOutputs，_schedule_prefill/_schedule_decode/_dp_allreduce
│       ├── block_manager.hpp   paged KV 的块分配
│       ├── zos.cpp/.hpp        zero-overhead scheduling（异步流水）
│       ├── mtp_varlen.hpp      MTP 变长验证
│       ├── prepare_sample.cpp  采样前处理（含 GPU 上的动态长度决策落地）
│       └── process_output.cpp
├── deepinfer/
│   ├── engine/llm_engine.py    45 KB，主循环：step / _execute_model_async_step / post_process / do_save_kvcache
│   ├── engine/arg_utils.py     所有启动参数的真相
│   ├── worker/cache_engine.py  49 KB，KV 分配与 state cache（block 与 SWA 都在这里）
│   ├── worker/model_runner.py  29 KB，单步前反向 + graph 回放
│   ├── worker/cuda_graph_runner.py, dflash_cuda_graph_runner.py
│   ├── worker/decode_perf_cache.py   拟合 cost(q,c) 的那个二次模型
│   ├── model_executor/models/deepseek.py      24 KB，V4.1 主体
│   ├── model_executor/models/v4_modules/attn.py        46 KB
│   ├── model_executor/models/v4_modules/index_attn.py  26 KB（Indexer）
│   ├── model_executor/models/v4_modules/yoco.py        YOCO / SWA
│   ├── model_executor/models/v4_modules/moe.py, mlp.py, cross_layer.py, engram.py, hc.py
│   ├── model_executor/models/mtp/dflash.py            DSpark draft + confidence head
│   ├── model_executor/layers/tilelang_ops/   ★ 每个算子三份：根目录 / cuda/ / asc/
│   ├── eplb/                   policy.py（算）、shuffle.py（搬）、__init__.py（接）
│   ├── structure_output/       语法自动机 + mask
│   └── environ.py              全部 env 变量与默认值（读这个比读文档快）
├── deep_cluster/               集群侧
├── doc/inference_deployment_qa.md   ★ 先读这个
├── examples/                   offline_inference.py、refresh_offline_decode_ppl.py、sweep/
└── tests/                      PPL 基线 + 各算子一致性测试
```

**推荐阅读顺序（约 3 小时）**：

1. `README.md` → `doc/inference_deployment_qa.md`（1 小时，把部署形态和数字记住）。
2. `deepinfer/environ.py`（15 分钟，看他们关心哪些旋钮）。
3. `csrc/engine/scheduler.hpp` + `zos.cpp`（40 分钟，调度与异步流水的本体）。
4. `deepinfer/worker/cache_engine.py`（40 分钟，block KV 与 SWA state 的分配；对照 QA 文档第 5 节读）。
5. `model_executor/layers/tilelang_ops/schedule_mtp_varlen.py` + `sampler.py`（30 分钟，看一个算子怎么写成三份实现、怎么测）。
6. `examples/README.md`（15 分钟，记住 benchmark 口径）。

---

## 5. 数字口径：他的数字在你的机器上不能直接用

**benchmark 默认**（`examples/README.md`）：`decode_num_requests=64`、`context_length=32768`、`non_cached_tokens=4096`、`prefill_batch_size=32768`、nextn=5（`DEEPINFER_USE_NEXTN_DECODE=0` 关闭）。

**指标定义**（这几个定义决定了数字能不能横向比）：
- prefill throughput：tokens/s/GPU，**跳 5 个 warmup chunk，取 10 个稳态区间**。
- decode `avg_SPS`：跳过 50 步 warmup + 60 步 logged，取 60 个区间，**剔除与 MoE LB 统计 flush 冲突的区间**，另报 `max_step`。
- decode throughput = `avg_SPS × decode_num_requests × (1+nextn) / --cp`。这是**全接受口径**（假设投机解码全部接受），不是实测接受率。和别人比 SPS 时要先确认对方用的是哪个口径。

**两个必须知道的坑**：
1. `PROFILE=1` 时步内含 profiler 开销，**bsz=1 可虚高近 50%**，不能拿来推 SPS；profiler 退出后仍有 **7–11% 残留减速**。
2. `ep_size = WORLD_SIZE × len(CUDA_VISIBLE_DEVICES)`，且 `--cp` 必须整除 `ep_size`——参数有整除约束，不是随便填。

PPL 基线 `tests/models/offline_decode_ppl.json` 是**按 platform/workload 键的逐样本精确比对**，刷新脚本会写锁文件 `tests/models/offline_decode_ppl.json.lock`（多节点）。也就是说他们用"逐样本 PPL 完全一致"当正确性判据，比"输出几个 token 一样"严格得多。

---

## 6. 与本项目的关系（哪些能借鉴，哪些不能）

**能直接借鉴的方法论**：
1. **用结构换常数**（SWA bounded replay）：先证明"这部分重算对结果没贡献"，再删掉，而不是优化它。我们测 decode 步开销时也可以用同一问法：这一步里哪些工作对结果没有贡献？
2. **删除同步而不是加速计算**（异步调度）：4.8 ms 的 decode 步里，如果有一段是 CPU-GPU 串行等待，它的代价是两段之和。这是下一步值得量的东西。
3. **可复现性是设计出来的**（batch-invariant + 确定性采样）：我们做 A/B 时踩过"采样链路本身不可逐位复现"，他们的答案是把这个自由度从系统里去掉，而不是在测量端补偿。
4. **统计系统本身会拖慢系统**（MoE LB flush 要剔除区间）：和我们"诊断开销与无诊断性能分开"是同一条纪律。
5. **把编译移出推理路径**（结构化输出）：任何"第一次请求要编译/加载"的东西，都应该在请求到达前做完。

**不能对标的东西**：
1. **规模**：64–128 卡超节点、100 PiB SSD、155 万亿 token/天。本机单卡 8 GB，任何"他们的吞吐是我们的 X 倍"的句子都没有意义。
2. **Wide-EP 收益**：EP64 的收益来自"每卡只读 1/64 专家"，前提是有 DeepEP 和超节点互联。单卡上这个话题不存在。
3. **min-p**：`min_p` 只出现在 `worker/model_runner.py` 与 `hugging_face/processor.py`，不是核心算子。我们那个 PR（#58907）在这里没有对应物，不要硬套。
4. **GDR / 3FS / EPLB 的绝对数**：这些数依赖具体硬件（400 Gbps、NVLink 拓扑、SSD 型号）。能引用的是"为什么这么选"，不是"多快"。

---

## 7. 未开源的部分（决定了你复现不出什么）

README 明列三项：
1. **KV cache 系统**（跨机前缀缓存、3FS 集成、Entrypoint 的 hash-chain 匹配）。
2. **VE / Prefill / Decode 的进程级分离**（引擎内部大部分已实现，缺的是把它们拆成独立服务的那层）。
3. **prefill/decode 各自的 DP 副本负载均衡**（`_dp_allreduce` 在代码里能看到雏形，但完整策略与服务端配额管理不在）。

再叠加一条文档里的话：内部 kernel 库与开源版"大体相同"，但 **FlashMLA 差一个手调 SASS 优化**，使 decode kernel 慢 5%–10%，**端到端差 ≤5%**。这句话本身是个很好的口径示范：kernel 级 10% 的差距，到端到端只剩 5%——因为 decode 步里还有采样、调度、通信。

所以：**你可以复现单机离线推理与 kernel 级数字，复现不了"生产环境 +4–7% 吞吐"这类端到端结论**。后者需要他们的集群。

---

## 8. 自测（读完能不能讲清楚）

1. 为什么 decode 用 EP 而 attention 用 DP，而 prefill 反而用小的 EP？各自由什么资源限制？
2. KV 为什么放"专用存储集群"而不是"分层存储 + 前缀亲和"？三条理由分别对应什么故障/成本？
3. SWA KV 为什么只存 128 token 整倍数位置，以及为什么最大命中长度是 L−134？
4. bounded replay 为什么"与上下文长度无关"？它的注意力窗口截断在哪一步发生？
5. 异步调度里，为什么 schedule(N+1) 不需要第 N 步的 confidence？如果坚持要，会退回什么状态？
6. DSpark 在低并发（16）为什么是负收益？
7. `moe_lb = 1.090` 是什么量？为什么 benchmark 要剔除某些区间？
8. decode throughput 的公式里为什么有 `(1+nextn)`，这意味着什么口径？
9. batch-invariant 推理放弃了什么自由度？换来的是什么？
10. 为什么"前缀匹配"这一步的开销藏不掉，而"取 KV"藏得掉？

（答案都在上文对应小节；如果某题答不出来，说明那节还需要再读一遍。）
