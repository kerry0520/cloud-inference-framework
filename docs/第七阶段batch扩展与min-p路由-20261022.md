# 第七阶段：batch 1→32 的收益曲线与 8 行路由悬崖（2026-10-22）

第五阶段只测到 batch 8，并用“分子固定 ≈56.9 µs ÷ 越来越长的 decode 步”解释百分比随 batch
下降。那个模型默认分子恒定，但从没在 batch>8 上验证过。本轮把 batch 推到 32，同时回答两件事：
收益是否继续按 1/step 稀释，以及分子是否仍然是个常数。

## 1. 方法

- 引擎内配对 ABBA（`scripts/abba_patch.py`）：同一进程里两臂常驻，切换只是一次
  `collective_rpc` 的标志翻转，测量不付导入成本。
- batch = 1, 2, 4, 8, 16, 32；**新增 `--max-num-seqs`（本轮用 32）**，脚本默认仍是 8。
  必须 ≥ 最大 batch，否则多出来的请求会排队，decode 间隔就不再代表该 batch 的一步。
- prompt 128 个合成 token、`max_tokens=64`、`min_p=0.1`、`temperature=0.8`、每请求固定 seed、
  prefix caching 关闭、fp16、`gpu_memory_utilization=0.65`。
- 每 batch 24 轮，每轮 ABBA／BAAB 各一次，共 4 次调用；每轮产生一个配对差，区间是**轮级**
  bootstrap（不是请求级）。
- 等价相位每 batch 4 次调用（base/patched/patched/base）比较 token ids，只记录不断言。
- 内核路由由 worker 的启动计数给出，用来判断实际跑的是哪条路径。

复现命令：

```bash
bash scripts/run_stage.sh abba-scale32          # 新增 stage，等价于下面的显式参数
# 或：run_stage.sh abba-scale --rounds 24 --batches 1,2,4,8,16,32 --max-num-seqs 32
```

## 2. 结果（`results/abba-scale-20261022T122837Z-HKWPoj/`）

| batch | decode ms/步（base → patched） | 每步省 µs（配对中位） | decode Δ%（中位，95% CI） | wall Δ% | min-p 调用/步 | 实际内核 |
|---|---|---|---|---|---|---|
| 1 | 4.09 → 4.04 | +53 | −1.27（−2.09…−1.15） | −1.31 | 1.02 | parallel |
| 2 | 4.26 → 4.29 | +39 | −0.91（−1.23…−0.29） | −0.76 | 1.03 | parallel |
| 4 | 4.63 → 4.57 | +48 | −1.07（−2.03…−0.70） | −1.03 | 1.03 | parallel |
| 8 | 4.72 → 4.71 | +36 | −0.77（−0.95…**+0.40**） | −0.56 | 1.03 | parallel |
| 16 | 5.55 → 5.54 | −16 | +0.30（−2.66…+0.85） | −0.00 | 1.03 | upstream fallback |
| 32 | 7.05 → 7.01 | +18 | −0.24（−0.67…+0.05） | −0.02 | 1.05 | upstream fallback |

读法：

- batch 1/2/4 的区间排除 0，方向与历史两轮一致；batch 8 本轮区间**含 0**
  （历史 run B 是 −0.91 [−1.55, −0.75]），所以这一档只能说“方向一致、本轮不显著”。
- batch 16/32 的区间跨 0，**收益归零**：不是变小，是没有发生。
- 每步省下的绝对量：窗口内 36–53 µs，和历史推得的 ~56.9 µs 同一量级。

## 3. 为什么 16/32 归零：8 行路由悬崖

补丁里 `PARALLEL_MAX_ROWS = 8`（`patches/mrv2_min_p.py`）：只有 `num_tokens <= 8` 且
`vocab >= 32000` 才走两阶段并行内核，否则回落上游“一行一个 CTA”的实现。器测数据对上了：

- batch ≤ 8：每步 1 次 min-p 调用，全部落在 parallel 双内核上（`_min_p_partial_max` +
  `_min_p_apply_threshold`，上游 `_min_p_kernel` 计数为 0）。
- batch 16/32：每步仍然只有 1 次 min-p 调用，但行数是 16/32，超过窗口 → 回落上游内核；
  只有收尾那几个不满行的步（rows ≤ 8，共 48 次调用）走 parallel。
- 也就是说：**batch 再大，快速内核根本不会执行**。第五阶段的“固定分子”只在 1–8 行成立，
  越过窗口后分子是 0，不是 56.9 µs。

顺带修正一条旧观察：PR #58907 交叉验证里写的“MRV2 把采样拆成每 8 行一组调用”，在这个版本
这条路径上不成立——batch 16 的调用形状是 `(16, 151936)`，一次调用处理全部行。那条观察来自
另一条采样路径，不能套用到这里。

## 4. 为什么窗口设在 8 行（不是拍脑袋）

上游 kernel 一行一个 CTA：batch 8 只有 8 个 CTA，RTX 5060 Laptop 的 SM 大量空闲，瓶颈是
“没活干”而不是带宽；两阶段把每行切成 37 个 4096 宽的分片，8 行就能发出 8×37=296 个 CTA，
把 SM 填满。batch 32 时上游已经有 32 个 CTA，接近填满，再切分只剩下调度与归约开销：

- 微基准（fp32、V151936）：batch 8 是 123.9 → 32.7 µs（3.8×），batch 32 是 161.1 → 154.9 µs
  （1.04×，见 `results/minp-micro-20261017T070433Z/timing.jsonl`）。
- 所以收益窗口由“SM 是否空闲”决定，而不是 batch 本身；窗口内还会被变长的 decode 步稀释
  （1.27% → 0.77%），窗口外直接归零。

## 5. 失败的第一次运行（保留）

第一次跑同一命令在 batch 16 断言失败：`assert kernels["_min_p_kernel"] == 0`。查计数发现
patched 臂在 batch 16 上有 3072 次上游内核启动——这正是路由设计的预期行为，是**脚本假设错了**，
不是补丁错了。已把断言改成按 batch 与内核名判断（窗口内要求 parallel 且上游为 0；窗口外要求
上游 >0 并记录 parallel 的收尾次数），重跑得到上面的结果。失败目录
`results/abba-scale-20261022T121849Z-wneYXT/`（含 `status.json` 的 traceback 与 1.2 MB
`timing.jsonl`）按纪律保留，未覆盖。

## 6. 等价性与已知噪声

- token id：batch 1/2/4/8/16 的四次调用全等；batch 32 有一次 patched 调用在第 16 个 token 上
  差 1 个 token。batch 32 两臂跑的是**同一个上游内核**，这个漂移只能来自已知的采样非确定性
  （同一 arm 两次调用也可能差 1 个 token），不能算到补丁头上。逐字节等价仍由 `verify` 模式与
  历史 96 组对照保证。

## 7. 限制

- 单机、单模型、fp16、合成 prompt、每 batch 24 轮；未锁频；绝对时间不可跨会话比较。
- wall 含 prefill；batch 32 的 prefill 会被 `max_num_batched_tokens=2048` 分块，wall 因此更钝，
  只能当旁证。
- 没有测“把窗口扩到 16/32 行”的引擎效果；微基准的 fp32 数字只能定性（引擎跑 fp16）。

## 8. 下一步

1. 若想把 batch>8 也纳入收益，需要在 16/32 行下重新设计并度量（微基准 32 行只有 4%，
   先证明内核本身值得，再改 `PARALLEL_MAX_ROWS`）——这是新补丁，要用户逐行审阅。
2. 结论可直接用于叙事：**“batch 越大收益越小”要拆成两段——窗口内按 1/step 稀释，
   窗口外因路由回落而归零；后者不是优化变弱，是这条实现只覆盖小 batch。**
