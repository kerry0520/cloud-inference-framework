# 云端推理框架：小批量 Decode 执行开销

在固定 vLLM 版本上研究小批量生成的输入准备、GPU 执行与结果处理，寻找可验证的专项改进。主线已完成环境、batch1/2/4/8 基线、两组 Decode 时间线、第一层线性投影的真实形状观测，以及 batch2 编译路径的逐层算子归因（GEMM 占单步内核时间 83.8%，lm_head 单独占 20.4% 且在 CUDA 图外）；小算子副目标已有受控接入候选。

副目标已收敛为可审查补丁：`patches/0001-mrv2-min-p-two-stage.patch`（147 行纯增量）把 Triton min-p 两阶段过滤打进 pinned 上游的 `vllm/v1/worker/gpu/sample/min_p.py`，派发窗口为 rows 1–8 且 vocab≥32000。补丁通过 96 组逐字节对照、边界探针、真实 MRV2 logits 4 次一致，且真实路径确实启动了新内核（`_min_p_kernel` 0 次）；见[第四阶段记录](docs/第四阶段min-p框架补丁-20261017.md)。大词表小批量的局部计时改善明显，整请求仅小幅变化，不能当作端到端提速。

- 自有脚本和结果入口：本目录。
- WSL 运行环境：`/opt/cloud-inference-framework`；上游源码独立位于其 `vllm-src/`，未修改。
- 初选版本：vLLM `0.31.0`，源码提交 `db9527a46873454610df6dbedf79a36d6bf1a7f6`。
- 模型：Qwen/Qwen3-0.6B，revision `c1899de289a04d12100db370d81485cdf75e47ca`。
- 学习入口：[连续学习记录](docs/连续学习记录.md)。
- Agent 接手入口：[当前交接](docs/当前交接.md)；每轮先读 AGENTS.md，结束前更新交接并保存本地 Git 检查点。
- 当前结果与下一步：[第一阶段记录](docs/第一阶段结果与续做-20261017.md)；逐层归因与带宽对照见[第三阶段记录](docs/第三阶段逐层归因与带宽对照-20261017.md)；min-p 补丁见[第四阶段记录](docs/第四阶段min-p框架补丁-20261017.md)与 `patches/README.md`。
- 运行方法：[环境与复现](docs/环境与复现.md)。

准备环境与结果以阶段记录为准，不把安装成功写作推理通过，不把本机结果推广到服务器。基线脚本的 wall 时间包含 prefill 和引擎处理；带 profiler 的记录不用于速度承诺。
