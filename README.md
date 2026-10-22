# 基于 vLLM 的小批量 Decode 性能分析与采样内核优化

本项目在固定 vLLM 0.31.0 / Qwen3-0.6B 上分析执行成本，并优化真实 MRV2 路径的 Triton min-p 算子。目录名“云端推理框架”沿用项目名；当前交付是框架性能研究与补丁。

## 已完成

- batch1/2/4/8基线、编译路径时间线、batch2逐层归因：一条trace中全部GEMM占83.8%，其中lm_head占20.4%，层内四类合计约63.4%。支持权重流成本重要，不证明达到硬件带宽上限。
- 两阶段min-p补丁：FP32、V151936、batch2局部122.9→14.0 µs（约8.8×）；两轮独立会话请求平均decode间隔配对变化约−1.10%／−1.00%。真实logits、边界和内核调用验证见阶段记录。min_p=0不受益。
- 两行启动参数版：减少维护成本，本机batch1～8的配对decode变化约−0.8%；部分wall区间跨0，不能把decode变化写成所有请求指标的稳定提速。
- batch 1→32 收益曲线（第七阶段）：窗口（≤8 行）内 decode 变化 −0.77%～−1.27%、每步省 36–53 µs；batch 16/32 超过补丁的 `PARALLEL_MAX_ROWS=8`，回落上游内核，收益归零（区间跨 0）。"固定节省 ÷ 变长的步"只解释了窗口内的稀释，窗口外是悬崖。
- 现成FP8配置与上下文扫描：FP8观察到约24%～27%平均decode间隔下降；8k上下文成本明显增加。FP8同时改变精度、内核与融合，不是本项目自研量化。旧−40%预测、跨臂相减得到的0.468×、“增量即attention占比”等说法已撤回，见第六阶段修订。

上述均为RTX5060Laptop/WSL、单模型、合成prompt、本机已测形状结果；第二设备、真实任务质量、多卡和生产服务尚未验收。profiled内核累计时间、局部微基准、请求平均decode间隔和wall是不同口径。

## 入口

- 接手：先读[项目总览与移交](docs/项目总览与移交-20261022.md)（三条线、数字口径、地图、下一步），再读 [AGENTS.md](AGENTS.md)、[当前交接](docs/当前交接.md)。
- 面试材料：[简历与面试口径](docs/简历与面试-20261020.md)、[项目叙事](docs/项目叙事-面试版.md)；学习入口：[教学手册](docs/教学手册-把一次decode拆到内核.md)、[连续学习记录](docs/连续学习记录.md)。
- 结果：[第一阶段](docs/第一阶段结果与续做-20261017.md)、[第二阶段](docs/第二阶段形状归因与min-p候选-20261017.md)、[第三阶段](docs/第三阶段逐层归因与带宽对照-20261017.md)、[第四阶段](docs/第四阶段min-p框架补丁-20261017.md)、[第五阶段](docs/第五阶段端到端配对与带宽模型-20261017.md)、[第六阶段](docs/第六阶段量化干预与上下文影响-20261018.md)。
- 补丁：[patches/README.md](patches/README.md)、[上游可行性记录](docs/上游PR可行性-20261018.md)、[PR #58907 交叉验证](docs/PR58907交叉验证-20261019.md)。
- HF3FS 独立修复：[vLLM PR #62393](https://github.com/vllm-project/vllm/pull/62393)、[分配修复与发布检查](docs/HF3FS分配修复与发布检查-20261019.md)。混合 hit/miss 的错误分配已复现并修复；18 个元数据测试、真实 HTTP 和受控 GPU 保存／加载验证通过。用户确认本人审阅与测试后已发布 PR；尚未合并。满池淘汰：[设计](docs/HF3FS淘汰设计-20261020.md)与租约+LRU 最小实现，26 个元数据测试与真实 HTTP 探针通过；已发布为 **[PR #62693](https://github.com/vllm-project/vllm/pull/62693)**（ready for review，stacked 在 #62393 上，head 提交 `c9b2b40`）；DCO 已通过，`pre-run-check` 是仓库策略门禁（新账号无已合并 PR、需维护者加标签，脚本明确禁止 AI agent 索要标签），因此上游 `pre-commit`/`Check format` 被跳过，本地 ruff 与测试已自查通过。**第三方已在两台 H100 + 真 3FS 上验证**：池满后保存继续成功、租约键 29/29 存活、过期租约 134 s 回收、28 次检查 0 错答案、保存/加载失败计数均为 0（见 [PR 描述草稿](docs/HF3FS淘汰PR描述草稿-20261020.md) 的验证表）；未解释的只有一次更大并发 flood 后的“最新 3 个找不到”。2026-10-22 上游 main 把 orjson 改成硬依赖，本 PR 因此与 main 冲突；已 rebase 到 `87b804f`（head `3ce1978`，26 passed + HTTP 探针通过，diff 仍 3 文件 +437/−28）并推送，GitHub 侧 `mergeable=true`。
- 环境：[本机复现](docs/环境与复现.md)、[远程迁移](docs/远程GPU租用与迁移-20261018.md)。远端安装尚无真实租用机全流程验收；不再提供未闭合的torch-only路径。
- 本次复核：[修复记录](docs/复核修复与验收-20261018.md)。

## 固定对象与证据

主项目位于Windows本目录；WSL runtime默认 `/opt/cloud-inference-framework`，支持 `CI_RUNTIME_DIR`。上游源码 `vllm-src/` 固定 `db9527a46873454610df6dbedf79a36d6bf1a7f6`，未修改；模型revision `c1899de289a04d12100db370d81485cdf75e47ca`。

项目远端为 <https://github.com/kerry0520/cloud-inference-framework>。已向上游 PR #58907 发过三条数据评论（最近一条是应作者邀请对 commit `95e9d21` 的本机交叉验证，见[交叉验证记录](docs/PR58907交叉验证-20261019.md)）。另已在用户明确授权并确认本人审阅／测试后发布 [HF3FS 分配修复 PR #62393](https://github.com/vllm-project/vllm/pull/62393)，尚未合并。一般变更先保存本地 Git，对外发布按用户授权执行。

原始结果及manifest在results/；Git忽略的trace／张量需另行复制和核对哈希。部分早期失败原件缺失，记录明确说明，不声称全部齐全。每轮结束更新交接与Git；工程已完成不代表用户已理解。
