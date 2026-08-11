# MLA遗留问题

## MLA Naive 与 Absorb 的选择

以 Batch Size 为 1、序列长度为 64K 的推理场景为例，Lightning Indexer 为每个 Query Token 选择 TopK=2048 个 KV Token。MLA 有以下三种实现方案。

### 方案一：MLA Naive + Sparse Mask

Prefill MLA 使用 Naive 模式。每个 Query Token 与所有历史 KV Token 计算 Attention，仅在 Softmax 前通过 Attention Mask 过滤 TopK 以外的 Token。两个批量矩阵乘的 Shape 如下。

| 计算            |   Batch |   M |   K |   N |
| --------------- | ------: | --: | --: | --: |
| BMM1（Q x K^T） | 1 x 128 | 64K | 192 | 64K |
| BMM2（P x V）   | 1 x 128 | 64K | 64K | 128 |

该方案的计算量与 Full Attention 相同，无法获得 DSA 的稀疏计算收益，不适合长序列场景。

### 方案二：MLA Naive + Sparse Attention

Prefill MLA 使用 Naive 模式，每个 Query Token 仅与 TopK=2048 个 KV Token 计算 Attention。由于每个 Query Token 独立选择 KV Token，序列长度 64K 被外提到 Batch 轴，BMM 的 M 轴为 1。

| 计算            |         Batch | M |    K |    N |
| --------------- | ------------: | -: | ---: | ---: |
| BMM1（Q x K^T） | 1 x 64K x 128 | 1 |  192 | 2048 |
| BMM2（P x V）   | 1 x 64K x 128 | 1 | 2048 |  128 |

该方案的 BMM 计算量约为方案一的 `2048 / 64K = 1/32`，但存在两个问题：

- BMM 的 M 轴为 1，矩阵乘计算效率较低。
- 每个 Query Token、每个 Attention Head 都需要独立读取选中的 KV，KV 数据难以跨 Query 和 Head 复用，BMM对kv的HBM访存量相较原始的Full Attention激增`topk=2048`倍，容易形成 HBM 访存瓶颈。

### 方案三：MLA Absorb + Sparse Attention

Prefill MLA 使用 Absorb 模式，与 Decode 保持一致。每个 Query Token 同样只与 TopK=2048 个 KV Token 计算 Attention，但 128 个 Query Head 共享同一份 latent KV。

| 计算            |   Batch |   M |    K |    N |
| --------------- | ------: | --: | ---: | ---: |
| BMM1（Q x K^T） | 1 x 64K | 128 |  576 | 2048 |
| BMM2（P x V）   | 1 x 64K | 128 | 2048 |  512 |

与方案二相比，方案三具有以下特点：

- BMM1 的 K 轴由 192 增至 576，BMM2 的 N 轴由 128 增至 512，总计算量约增加至 `1088 / 320 = 3.4` 倍。
- BMM 的 M 轴由 1 增至 128，更有利于发挥矩阵乘单元的计算效率。
- KV 从逐 Head 展开的 K/V 变为各 Head 共享的 latent KV，显著降低 HBM 读取量。

按 BMM1 和 BMM2 分别从 HBM 读取一次 KV 估算，方案二与方案三每个 Query Token 的 KV 元素读取量分别为：

```text
方案二：128 x 2048 x (192 + 128)
方案三：      2048 x (576 + 512)
```

两者之比为：

```text
[128 x 2048 x (192 + 128)] / [2048 x (576 + 512)]
= 128 x 320 / 1088
= 37.65
```


## FAQ 遗留问题

### 1. 什么时候使用矩阵吸收？什么时候使用非吸收？

MLA 矩阵吸收与非吸收在数学上等价，差异在于 KV 上投影矩阵位于 Attention 的哪一侧执行。

计算量对比

定义：

| 符号    | 含义                                     |
| ------- | ---------------------------------------- |
| `H`   | Attention Head 数                        |
| `Q`   | Query Token 数                           |
| `S`   | KV 序列长度或每个 Query 实际选择的 KV 数 |
| `d_c` | KV latent 维度                           |
| `d_k` | NoPE Key 维度                            |
| `d_r` | RoPE 维度                                |
| `d_v` | Value 维度                               |

忽略 Softmax、归一化和索引开销，两个 BMM 的主要计算量为：

```text
Naive BMM：  2 x Q x S x H x (d_k + d_r + d_v)
Absorb BMM： 2 x Q x S x H x (2 x d_c + d_r)
```

吸收模式还需要对 Query 和 Attention 输出执行上投影：

```text
Absorb 投影：2 x Q x H x d_c x (d_k + d_v)
```

非吸收模式需要将 latent KV 上投影为完整 K/V：

```text
Naive 投影：2 x S x H x d_c x (d_k + d_v)
```

在 Prefill 中，该投影通常对每个 KV Token 执行一次；在 Decode 中，若缓存完整 K/V，则只需对新增 Token 执行。若仅缓存 latent KV 并在每个 Decode Step 临时恢复完整 K/V，则该投影会对全部历史 KV 重复执行。

当 `d_c > d_k, d_v` 时，吸收模式的 Attention BMM 计算量更大。以 LongCat-2.0 的 `d_c=512`、`d_k+d_r=192`、`d_v=128` 为例，两个 BMM 的维度比为：

```text
(2 x 512 + 64) / (192 + 128) = 1088 / 320 = 3.4
```

对于稠密长序列 Prefill，`Q` 和 `S` 都较大，Attention 的二次项占主导。若标准 Naive Attention 算子效率较高且内存可承受，非吸收模式通常具有更低的理论计算量。

对于 Decode，`Q` 通常为 1，而 `S` 随上下文长度增长。吸收模式只对当前 Query 和输出执行投影，避免对全部历史 latent KV 重复展开 K/V，通常具有明显优势。

#### 搬运量对比

非吸收模式若缓存逐 Head 展开的完整 K/V，且 RoPE Key 也随 Head 展开，则每个 Token 的 KV Cache 元素数为：

```text
H x (d_k + d_r + d_v)
```

吸收模式缓存 latent KV 与独立 RoPE Key，每个 Token 的元素数为：

```text
d_c + d_r
```

两者的完整展开表示及理想读取量之比为：

```text
H x (d_k + d_r + d_v) / (d_c + d_r)
```

以 `H=128`、`d_c=512`、`d_k=128`、`d_r=64`、`d_v=128` 为例：

```text
[128 x (128 + 64 + 128)] / (512 + 64) = 71.1
```

在上述布局下，吸收模式可将 KV Cache 容量和理想 KV 读取量降低约 71 倍。如果 RoPE Key 在各 Head 间共享，则非吸收模式每个 Token 的元素数为 `H x (d_k + d_v) + d_r`，对应比例约为 `57.0` 倍。如果非吸收模式仍只缓存 latent KV，则每个 Decode Step 都需要重新展开历史 K/V，虽然静态 Cache 较小，但会引入与 `S` 成正比的额外计算和中间数据搬运。

#### Roofline 分析

Roofline 模型使用算术强度（Arithmetic Intensity，AI）统一衡量计算量与 HBM 搬运量：

```text
AI = FLOPs / HBM Bytes
可达性能 = min(P_peak, BW_eff x AI)
拐点算术强度 AI_ridge = P_peak / BW_eff
```

其中，`P_peak` 为目标数据类型的峰值计算性能，`BW_eff` 为算子实际可用的 HBM 带宽。若 `AI < AI_ridge`，算子主要受 HBM 带宽限制；若 `AI > AI_ridge`，算子主要受计算能力限制。

以 LongCat-2.0 的稀疏 Prefill 为例，令 `b` 表示每个 KV 元素的字节数，仅统计 BMM1、BMM2 的主要 FLOPs 和 KV 读取量。

Naive 模式中，每个 Query Head 独立读取展开后的 K/V：

```text
FLOPs_naive = 2 x H x S x (d_k + d_r + d_v)
Bytes_naive = b x H x S x (d_k + d_r + d_v)
AI_naive    = 2 / b
```

BF16 的 `b=2`，因此：

```text
AI_naive = 1 FLOP/Byte
```

Absorb 模式将 `H` 个 Head 放在 BMM 的 M 轴，并共享同一份 latent KV：

```text
FLOPs_absorb = 2 x H x S x (2 x d_c + d_r)
Bytes_absorb = b x S x (2 x d_c + d_r)
AI_absorb    = 2 x H / b
```

当 `H=128`、数据类型为 BF16 时：

```text
AI_absorb = 128 FLOPs/Byte
```

该估算忽略了 Query、输出、Attention Score、中间结果和索引的搬运，因此不是完整算子的实测 AI，但能够反映两种模式的主要差异：Naive 的计算量较少，但 KV 按 Head 重复读取，算术强度低；Absorb 的 FLOPs 约增加至 3.4 倍，但主 KV 读取量约降低 37.6 倍，算术强度提高约 128 倍。

在 Roofline 图上，两种模式通常表现为：

| 模式                    | BMM Shape 特征                         | 算术强度 | 常见瓶颈                         |
| ----------------------- | -------------------------------------- | -------: | -------------------------------- |
| Naive Sparse Attention  | `M=1`，KV 按 Query、Head 独立 Gather |       低 | HBM 带宽、离散访存和矩阵乘利用率 |
| Absorb Sparse Attention | `M=H`，各 Head 共享 latent KV        |       高 | 可能从带宽受限转为计算受限       |

因此，不能仅根据 FLOPs 判断性能：

- 若 Naive 位于 Roofline 的带宽受限区，减少 FLOPs 对耗时帮助有限；Absorb 即使增加计算量，也可能因显著减少 HBM 流量而更快。
- 若 Absorb 仍位于带宽受限区，其理想收益主要由 HBM 搬运量下降决定；实际收益还取决于 Gather 的有效带宽和片上缓存复用。
- 若 Absorb 越过 Roofline 拐点进入计算受限区，继续减少 KV 搬运的收益会减弱，此时应重点优化 BMM 利用率和额外投影计算。
- 若 Naive 已具有较高的数据复用并进入计算受限区，Absorb 增加的 latent 维度和投影计算可能直接转化为额外耗时。

对应到推理场景：

| 场景                               | Roofline 特征                                             | 推荐模式                     |
| ---------------------------------- | --------------------------------------------------------- | ---------------------------- |
| 单 Token Decode、长上下文          | Query 数少，历史 KV 读取量大，Naive AI 低                 | Absorb                       |
| 稀疏 Prefill，KV 按 Query 独立选择 | Query 间难以复用 KV，Naive 的`M=1` 且离散访存           | Absorb                       |
| 稠密 Prefill                       | 同一 KV 可被多个 Query 复用，Naive AI 随 Query Block 增大 | 优先评估 Naive               |
| 短序列或 KV 可驻留片上缓存         | HBM 搬运占比下降，额外 FLOPs 更敏感                       | 以实测结果为准               |
| KV 量化                            | 单元素字节数下降，两种模式的 AI 均提高                    | 重新按量化数据类型计算并实测 |

实际选型时，应从 Profiling 数据获得 `FLOPs`、`HBM Bytes`、有效带宽和计算利用率，再与目标硬件对应数据类型的 `AI_ridge` 比较。稀疏 Gather 通常无法达到连续读取的峰值带宽，因此应使用实测 `BW_eff`，不能直接代入硬件标称带宽。

#### 选型建议

| 场景                                       | 推荐模式        | 主要原因                                               |
| ------------------------------------------ | --------------- | ------------------------------------------------------ |
| 单 Token Decode、长上下文或大 Batch Decode | Absorb          | 避免展开历史 K/V，显著降低 KV Cache 读取量             |
| MTP Decode，且 Query 数远小于 KV 长度      | Absorb          | 多个 Query 仍可复用 latent KV，访存收益通常占优        |
| 稀疏 Prefill，TopK KV 按 Query 独立选择    | Absorb          | 可跨 Head 共享 latent KV，并将 Head 维放入 BMM 的 M 轴 |
| 稠密长序列 Prefill                         | 优先评估 Naive  | Naive 的 Attention 维度更小，理论计算量更低            |
| 短序列、低 Batch                           | 以实测结果为准  | KV 搬运收益有限，额外投影和算子调度可能抵消收益        |
| Absorb 算子不支持目标维度、布局或量化模式  | Naive           | 首先保证算子兼容性和精度正确性                         |
| 需要统一 Prefill 与 Decode 路径            | 优先评估 Absorb | 可降低工程复杂度，但必须验证 Prefill 计算开销          |

总体原则如下：

- Decode 更关注历史 KV 的容量和 HBM 搬运，通常选择 Absorb。
- 稠密 Prefill 更关注 Attention 的计算量，通常优先评估 Naive。
- 稀疏 Prefill 需要同时考虑 BMM Shape 与 KV Gather 访存；当 Naive 路径的 M 轴过小且 KV 无法跨 Head 复用时，Absorb 可能更快。
- 最终选择应以目标序列长度、Batch Size、数据类型和实际 Profiling 结果为准，不能仅依据 FLOPs 判断。

### 2. 为什么 DeepSeek-R1 的 Prefill 使用 Naive、Decode 使用 Absorb，而 DeepSeek-V3.2-Exp 的 Prefill 和 Decode 均使用 Absorb？

这一差异不是 MLA 数学语义发生变化，而是 Attention 从稠密 Full Attention 演进为 DSA Sparse Attention 后，Prefill 的 BMM Shape、KV 复用方式和 Roofline 瓶颈发生了变化。

#### DeepSeek-R1：稠密 Prefill 优先减少计算量

DeepSeek-R1 的 Prefill 使用 `forward_page_attention_normal`。该路径先将 latent KV 上投影为完整 K/V，再执行稠密 Attention：

```text
latent KV -> W_UK/W_UV -> 完整 K/V -> Full Attention
```

对应代码位于 `models/deepseek_r1/models/modeling_deepseek.py`：

- `forward()` 在 `is_prefill=True` 时选择 `forward_page_attention_normal`。
- `forward_page_attention_normal()` 通过 `kv_b_proj_w_k` 和 `kv_b_proj_w_v` 生成完整 K/V。
- FA 的 `num_key_value_heads` 等于当前 Rank 的 Query Head 数，表明 Attention 使用逐 Head 展开的 K/V。

稠密 Prefill 中，一组连续 KV 可以被一个 Query Block 中的多个 Query 复用，BMM 的 M 轴为 Prefill Token 数，而不是 1。Naive 路径虽然需要执行一次 K/V 上投影，但 Attention 的计算维度较小：

```text
Naive：  BMM1 使用 d_k + d_r，BMM2 使用 d_v
Absorb：BMM1 使用 d_c + d_r，BMM2 使用 d_c
```

通常 `d_c > d_k, d_v`。当 Prefill 序列较长时，Attention 的二次计算项占主导，Absorb 会增加主要 BMM 的 FLOPs；而 Naive 的连续稠密 BMM 已具有较好的数据复用和矩阵乘效率。因此，R1 Prefill 选择 Naive，以较小的 Attention 维度降低计算量。

#### DeepSeek-R1：Decode 优先减少 KV 搬运

DeepSeek-R1 的 Decode 使用 `forward_page_attention_absorb`；开启 `enable_mla_prolog` 时，使用 `forward_page_attention_mla_prolog`，但仍由融合算子完成矩阵吸收。

Decode 通常每个请求只有一个 Query Token，历史 KV 长度则持续增长。如果使用 Naive，有两种实现方式：

- 缓存完整 K/V：避免重复上投影，但 KV Cache 容量和每步 HBM 读取量显著增加。
- 只缓存 latent KV：每个 Decode Step 都需要重新展开全部历史 K/V，引入与上下文长度成正比的额外计算和搬运。

Absorb 将 `W_UK` 移到当前 Query 侧，将 `W_UV` 移到 Attention 输出侧，只读取 latent KV。此时额外投影仅与当前 Query 数相关，而 KV 搬运收益随上下文长度增长。因此 R1 Decode 选择 Absorb。

#### DeepSeek-V3.2-Exp：DSA 改变了 Prefill 的性能瓶颈

DeepSeek-V3.2-Exp 引入 DSA。Lightning Indexer 为每个 Query Token 选择 TopK=2048 个 KV Token，随后执行 Sparse Flash Attention。仓内优化文档 `docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_inference_guide.md` 明确比较了三种 Prefill 方案：

| 方案                      | BMM Shape 特征                          | 主要问题或收益                             |
| ------------------------- | --------------------------------------- | ------------------------------------------ |
| Naive + Sparse Mask       | 仍计算 Full Attention                   | 无法获得稀疏计算收益                       |
| Naive + Sparse Attention  | Batch 包含 Query Token 和 Head，`M=1` | 矩阵乘效率低，KV 按 Query、Head 离散读取   |
| Absorb + Sparse Attention | Batch 只包含 Query Token，`M=128`     | 各 Head 共享 latent KV，HBM 读取量显著下降 |

Naive Sparse Attention 无法像稠密 Attention 一样让一个连续 KV Block 被多个 Query 高效复用。每个 Query Token 都有独立的 TopK 索引，且逐 Head 展开的 K/V 需要分别 Gather，形成大量离散 HBM 访问。其典型 BMM Shape 为：

```text
BMM1: Batch = 64K x 128, M = 1, K = 192,  N = 2048
BMM2: Batch = 64K x 128, M = 1, K = 2048, N = 128
```

Absorb 将 128 个 Head 放到 BMM 的 M 轴，并让这些 Head 共享同一份 latent KV：

```text
BMM1: Batch = 64K, M = 128, K = 576,  N = 2048
BMM2: Batch = 64K, M = 128, K = 2048, N = 512
```

按前文的 BF16 主 BMM 估算：

```text
Naive Sparse AI 约为 1 FLOP/Byte
Absorb Sparse AI 约为 128 FLOPs/Byte
Absorb FLOPs 约增加至 3.4 倍
Absorb 主 KV 读取量约降至 1/37.6
```

因此，V3.2-Exp 的 Naive Sparse Prefill 明显偏向带宽受限，且 `M=1` 不利于矩阵乘单元利用；Absorb 虽然增加 FLOPs，却显著提高算术强度和 BMM 的 M 轴利用率。在这一场景中，减少离散 KV 搬运比减少理论 FLOPs 更重要。

#### DeepSeek-V3.2-Exp：Prefill 和 Decode 统一使用 Absorb

代码实现与上述设计一致，见 `models/deepseek_v3_2_exp/models/modeling_deepseek.py`：

- 普通 Prefill 和 Decode 都进入 `forward_absorb()`，CP Prefill 进入 `forward_absorb_cp()`。
- `mlaprolog_prefill()` 和 `mlaprolog_decode()` 都向 `npu_mla_prolog_v3` 传入 `weight_uk=self.kv_b_proj_w_k`。
- Sparse Flash Attention 的 `key` 和 `value` 都使用 `k_nope`，即直接在 latent 空间计算。
- `mla_epilog(..., absorb=True)` 使用 `kv_b_proj_w_v` 将 latent Attention 输出映射回 Value 空间。

统一 Absorb 路径还带来以下工程收益：

- Prefill 和 Decode 复用相同的 latent KV Cache Layout。
- MLA Prolog、Sparse Flash Attention 和 MLA Epilog 的融合与量化路径保持一致。
- CP、KVCache Offload 和 C8 KVCache 无需维护另一套完整 K/V Attention 路径。

#### 结论

| 模型与阶段                | Attention 形态                           | 主要瓶颈                             | 选择                               |
| ------------------------- | ---------------------------------------- | ------------------------------------ | ---------------------------------- |
| DeepSeek-R1 Prefill       | 稠密 Full Attention                      | 大规模 BMM 计算                      | Naive，降低 Attention 维度和 FLOPs |
| DeepSeek-R1 Decode        | 单/少量 Query 读取长历史 KV              | KV Cache 容量与 HBM 带宽             | Absorb，避免展开历史 K/V           |
| DeepSeek-V3.2-Exp Prefill | 每个 Query 独立 TopK 的 Sparse Attention | 离散 KV Gather、`M=1` 和低算术强度 | Absorb，以更多计算换取更少搬运     |
| DeepSeek-V3.2-Exp Decode  | Sparse Attention 读取长历史 latent KV    | KV 搬运与稀疏访问                    | Absorb                             |

归根结底，R1 与 V3.2-Exp 的差异来自 Prefill Attention 的数据复用模式：R1 的稠密 Prefill 能够有效复用完整 K/V，Naive 更有利于降低 FLOPs；V3.2-Exp 的稀疏 Prefill 按 Query 独立选择 KV，Naive 会失去跨 Query、跨 Head 的数据复用，Absorb 更符合其 Roofline 特征。

### 3. Lightning Indexer 对整网性能有什么影响？部署 LI 后性能瓶颈如何变化？

Lightning Indexer（LI）是 DSA 的稀疏路由模块。它先使用轻量的类 MQA 结构对完整历史序列打分，为每个 Query 选择 `TopK=2048` 个 KV Token，再由 Sparse Flash Attention（SFA）仅对这些 Token 计算 Attention：

```text
完整历史 KV -> LI 全序列打分与 TopK -> SFA 读取 TopK KV -> Attention 输出
```

LI 的核心价值不是消除全序列扫描，而是使用更小的 Indexer Key 表示完成筛选，将计算和搬运开销更高的主 Attention 从全量 KV 限制到固定 TopK 范围。

#### 整网收益与新增开销

部署 LI 后，主 Attention 的复杂度由稠密计算近似降为：

```text
Prefill:
  Full Attention = O(S^2 x d_attn)
  LI             = O(S^2 x d_index)
  SFA            = O(S x TopK x d_attn)

Decode（每个新 Token）:
  LI             = O(S x d_index)
  SFA            = O(TopK x d_attn)
```

其中，`S` 为历史序列长度，`d_index` 为 Indexer 的轻量打分维度，`d_attn` 为主 Attention 的计算维度。LI 在 Prefill 阶段仍包含随 `S^2` 增长的计算，在 Decode 阶段仍需扫描长度为 `S` 的 Indexer Key Cache；但它将后续 SFA 的 KV 读取和 Attention 计算限制在 `TopK=2048`。因此，序列越长，稀疏化收益通常越明显。

LI 同时引入以下额外成本：

- 每层新增 `q_b_proj`、`wk_proj` 和 `weight_proj`，全模型约增加 `0.85B` 参数。
- Decode 需要额外缓存 Indexer Key，大小为 `batch_size x kv_length x indexer_head_dim x storage_bytes x num_layers`。仓内文档给出的示例中，每个 Rank 处理 4 Batch、序列长度为 64K、使用 BF16 时，新增缓存约为 4 GB。
- LI 包含 Score BatchMatmul、ReLU、ReduceSum 和 TopK；长序列下 TopK 本身会成为热点。

#### Prefill：瓶颈由主 Attention 转向 LI

引入 DSA 前，长序列 Prefill 的主要成本是 Full Attention 的大规模 BMM 和全量 KV 访问。引入 LI 后，SFA 只计算 TopK KV，主 Attention 的计算量和访存量显著下降；LI 对所有 Query 与历史 Token 进行打分，其计算量随 `S^2` 增长，因此成为长序列 Prefill 的主要瓶颈之一。

LI 还需要沿 Head 维执行 Reduce Sum。若直接使用 TP 切分 Head，该归约会引入较大的跨卡通信开销。仓内实现因此在 Prefill Attention 中采用 Context Parallelism（CP）切分序列，并通过头尾对称切片缓解因果 Attention 的负载不均。换言之，LI 不仅改变了算子热点，也改变了整网的优选并行策略。

#### Decode：瓶颈转向全序列筛选和离散 Gather

Decode 中，SFA 的 Attention 计算范围基本固定为 TopK，但 LI 每一步仍需读取完整 Indexer Key Cache 并完成全序列打分。因此，随着上下文增长，LI 的耗时和缓存带宽压力近似线性增长，其整网耗时占比会提高。

LI 输出的 TopK 索引还会使 SFA 从连续读取变为离散 Gather。此时 SFA 的主要瓶颈不再是 Attention FLOPs，而是从完整 KVCache 中聚合 TopK KV 的不规则访存。部署 LI 后，Attention 内部的主要热点可概括为：

```text
Full Attention 的大规模 BMM 与全量 KV 搬运
    -> LI 全序列打分、ReduceSum 与 TopK
    -> SFA 的 TopK KV 离散 Gather
```

由于 Attention 总耗时下降，MoE、普通 MatMul、MLA Epilog 中的升维和 `o_proj` 等模块在整网中的相对占比也会提高。此时继续只优化 Attention 的边际收益会下降，需要同步关注 MoE、矩阵乘和流水并行。

#### 从计算量和搬运量看瓶颈迁移

仓内文档给出了 64K 序列、4 Batch 下的估算：

| 模块 | 模式   | KV 搬运量 |  Cube 计算量 | `QK^T` 搬运量 |
| ---- | ------ | --------: | -----------: | --------------: |
| MLA  | 非 MTP |    144 MB | 19.33 GFLOPs |           32 MB |
| SFA  | 非 MTP |    4.5 MB |  0.60 GFLOPs |            1 MB |
| LI   | 非 MTP |     32 MB |  2.15 GFLOPs |           16 MB |

相较稠密 MLA，SFA 的 KV 搬运量由 144 MB 降至 4.5 MB，Cube 计算量由 19.33 GFLOPs 降至 0.60 GFLOPs；但 LI 仍需搬运 32 MB 的 Indexer Key，并执行 2.15 GFLOPs 计算。因此，主 Attention 被稀疏化后，LI 的耗时占比自然上升。该对比反映的是模型化的模块开销，不代表各模块耗时可按数值直接线性换算；TopK 和离散 Gather 的实际效率还取决于算子实现与有效 HBM 带宽。

#### 对 MTP 和 KVCache Offload 的影响

MTP 对 LI 和 SFA 的收益不同：

- LI 的 Indexer Key Cache 搬运量与 Query 数关系较弱。MTP1 可在近似相同的 Key 搬运量下增加计算量，提高算术强度，因此 LI 通常能从 MTP 获益。
- SFA 中每个 Query 独立选择 TopK。最坏情况下，MTP1 的两个 Query 没有 TopK 重叠，稀疏 KV 搬运量接近翻倍，因此 MTP 对 SFA 的加速有限。

LI 还为 KVCache Offload 提供了选择依据。仓内实现使用 LI 返回的 TopK 索引，通过 `GatherSelectionKvCache` 仅将命中的完整 KV 从 Host 搬到 Device；相邻 Decode Step 的 TopK 若有 60% 重合，只需新增搬运其余 40%。因此，LI 会增加 Indexer Key Cache，但也使完整 KVCache 的按需搬运和跨步复用成为可能。

#### 结论

| 影响项              | 部署 LI 前                    | 部署 LI 后                           |
| ------------------- | ----------------------------- | ------------------------------------ |
| 主 Attention        | 扫描并计算全量 KV             | SFA 仅计算 TopK KV                   |
| 长序列 Prefill 瓶颈 | Full Attention BMM 与 KV 搬运 | LI 的`S^2` 打分、ReduceSum 和 TopK |
| 长序列 Decode 瓶颈  | 全量 KV 读取与 Attention 计算 | LI 全序列扫描与 SFA 离散 Gather      |
| 并行策略            | 主要围绕 Attention 计算切分   | LI 的 Head 归约使 Prefill 更适合 CP  |
| Cache               | MLA KVCache                   | MLA KVCache 与 Indexer Key Cache     |
| 整网热点            | Attention 占比较高            | MoE、MatMul 和 Epilog 相对占比提高   |

总体而言，LI 使用较轻的全序列筛选换取主 Attention 的 TopK 稀疏化。它显著降低了长序列 SFA 的计算量和 KV 搬运量，但将性能瓶颈转移到 LI 的全序列打分与 TopK、SFA 的离散 Gather，以及 Attention 之外的 MoE 和 MatMul。仓内 128K Benchmark 中 DeepSeek-V3.2-Exp 吞吐达到 DeepSeek-V3.1 的 450%，这是 DSA、融合算子、量化、并行和流水等整套优化共同作用的结果，不能视为 LI 单模块的独立收益。

### 4. MLA 推理链路包含哪些算子？这些算子的作用和输入输出是什么？

下面以仓内 DeepSeek-R1 和 DeepSeek-V3.2-Exp 的模型实现为例。R1 展示了 MLA Naive 与 Absorb 两条路径，V3.2-Exp 展示了 Absorb MLA 与 LI、SFA 组合后的完整 DSA 路径。

#### 符号与默认维度

| 符号        | 含义                                   | DeepSeek-R1 / V3.2-Exp 默认值 |
| ----------- | -------------------------------------- | ----------------------------: |
| `T`       | 当前调用中所有 Query Token 数          |           随 Batch 和阶段变化 |
| `H`       | Hidden Size                            |                由模型配置决定 |
| `N`       | Attention Head 数                      |                           128 |
| `N_local` | 当前 Attention TP Rank 的 Head 数      |          `N / attn_tp_size` |
| `d_c`     | MLA KV latent 维度，即`kv_lora_rank` |                           512 |
| `d_n`     | 非 RoPE 的 Q/K Head 维度               |                           128 |
| `d_r`     | RoPE Head 维度                         |                            64 |
| `d_v`     | Value Head 维度                        |                           128 |
| `K_top`   | V3.2-Exp 每个 Query 选择的 KV 数       |                          2048 |

MLA 不缓存每个 Head 的完整 K/V，而是缓存两部分：

```text
nope_cache: latent KV，逻辑维度 [num_blocks, block_size, 1, d_c]
rope_cache: RoPE Key，逻辑维度 [num_blocks, block_size, 1, d_r]
```

其中 latent KV 不沿 Attention Head 展开，因此 `num_head=1`。分页缓存实际可采用 `PA_NZ` 或 `PA_BSND` 布局，但其逻辑含义不变。

#### 总体数据流

DeepSeek-V3.2-Exp 的 Absorb 路径可概括为：

```text
hidden_states [T,H]
    -> MLA Prolog
       -> q_nope [T,N_local,d_c]
       -> q_rope [T,N_local,d_r]
       -> 写入 nope_cache / rope_cache
    -> Lightning Indexer
       -> topk_indices [T,1,K_top]
    -> Sparse Flash Attention
       -> latent attention output [N_local,T,d_c]
    -> MLA Epilog: W_UV + o_proj
       -> attention output [T,H]
```

R1 Absorb Decode 不包含 LI，直接使用融合 Attention 扫描有效的 latent KVCache；R1 Naive Prefill 则先将 latent KV 上投影为逐 Head 的完整 K/V，再执行稠密 Attention。

#### 1. Q/KV 投影与缓存写入：`npu_mla_prolog_v3`

V3.2-Exp 的 Prefill 和 Decode，以及 R1 开启 `enable_mla_prolog` 后的 Decode，使用 `npu_mla_prolog_v3`。该算子融合了以下计算：

```text
Q 路径：x -> W_DQ -> RMSNorm -> W_UQ -> 拆分 Q_nope/Q_rope
                 Q_nope -> 与 W_UK 吸收 -> latent Query
                 Q_rope -> RoPE

KV 路径：x -> W_DKV_KR -> 拆分 C_KV/K_rope
                   C_KV -> RMSNorm
                   K_rope -> RoPE
                   C_KV/K_rope -> 写入分页 KVCache
```

主要输入如下：

| 输入                       | 典型 Shape 或类型                       | 作用                                                 |
| -------------------------- | --------------------------------------- | ---------------------------------------------------- |
| `token_x`                | `[T, H]`；CP Prefill 可为 `[1,T,H]` | 当前层输入 Hidden States                             |
| `weight_dq`              | Q 下投影权重                            | 将 Hidden States 投影到 Q LoRA latent                |
| `weight_uq_qr`           | Q 上投影权重                            | 生成各 Head 的 Q_nope 和 Q_rope                      |
| `weight_uk`              | `[N_local,d_n,d_c]` 等价布局          | 将`W_UK` 吸收到 Query 侧，使 Q_nope 映射到 `d_c` |
| `weight_dkv_kr`          | KV 下投影权重                           | 同时生成 latent KV 与 RoPE Key                       |
| `rmsnorm_gamma_cq`       | Q latent Norm 权重                      | 对 Q LoRA latent 执行 RMSNorm                        |
| `rmsnorm_gamma_ckv`      | `[d_c]`                               | 对 latent KV 执行 RMSNorm                            |
| `rope_sin`、`rope_cos` | `[T,d_r]` 等价布局                    | 对 Q_rope 和 K_rope 应用位置编码                     |
| `cache_index`            | `[T]`                                 | 每个新 Token 的物理写入 Slot，即`slot_mapping`     |
| `kv_cache`、`kr_cache` | 分页缓存 Tensor                         | 分别保存 latent KV 和 RoPE Key                       |
| 量化 Scale 与 Mode         | 可选                                    | 控制权重、Query 和 KVCache 量化                      |

主要显式输出为：

| 输出                     | 典型 Shape           | 作用                                                             |
| ------------------------ | -------------------- | ---------------------------------------------------------------- |
| `q_nope`               | `[T,N_local,d_c]`  | 已吸收`W_UK` 的非 RoPE Query，直接与 latent KV 计算得分        |
| `q_pe`                 | `[T,N_local,d_r]`  | 应用 RoPE 后的 Query 位置分量                                    |
| `dequant_scale_q_nope` | 与量化 Query 对应    | 量化 Attention 使用的反量化参数；非量化路径可忽略                |
| `qr`                   | `[T,q_lora_rank]`  | Q 下投影并归一化后的中间结果；V3.2-Exp 将其复用于 Indexer Q 投影 |
| `dequant_q_norm`       | 与量化 Q latent 对应 | Indexer 或后续量化投影使用的 Per-Token Scale                     |

该算子还有一个重要的原地副作用：根据 `cache_index` 将当前 Token 的 latent KV 与 RoPE Key 写入 `kv_cache` 和 `kr_cache`。因此，仅查看 Python 返回值会遗漏 MLA Prolog 的缓存更新职责。

R1 未启用融合 Prolog 时使用 `npu_kv_rmsnorm_rope_cache_v2` 完成 KV 路径：输入未归一化的 latent KV、RMSNorm 权重、RoPE `cos/sin`、`slot_mapping` 和两份 Cache；输出当前 Token 的 `k_nope`、`k_rope`（当 `is_output_kv=True`），同时写入分页缓存。Q 投影、Q RoPE 和矩阵吸收则由模型代码中的 Linear、RoPE 与 BatchMatmul 分别完成。

#### 2. 分页 KVCache 的写入与寻址

MLA Attention 使用两个容易混淆的索引：

| 输入                               | 使用阶段                     | 作用                                          |
| ---------------------------------- | ---------------------------- | --------------------------------------------- |
| `slot_mapping` / `cache_index` | MLA Prolog 或 Cache 更新算子 | 将当前 Token 写入指定物理 Slot                |
| `block_table`                    | FA、SFA 或 LI                | 将每个请求的逻辑 KV Block 映射到物理 Block    |
| `actual_seq_lengths_q`           | Attention/LI                 | 指明 Packed Batch 中每个请求的有效 Query 边界 |
| `actual_seq_lengths_kv`          | Attention/LI                 | 限制每个请求可读取的有效历史 KV 长度          |

`slot_mapping` 只负责写入位置，`block_table` 只负责读取寻址。两者必须来自同一套分页映射，否则即使 Shape 合法，也会读到其他请求或错误位置的缓存。

#### 3. R1 Naive Attention：展开完整 K/V 后计算

R1 的 `forward_page_attention_normal()` 用于 Prefill。`npu_kv_rmsnorm_rope_cache_v2` 得到当前 Token 的 latent KV 后，模型显式执行：

```text
K_nope = C_KV x W_UK -> [N_local,T,d_n]
V       = C_KV x W_UV -> [N_local,T,d_v]
K_rope  = 按 Head 复制 -> [N_local,T,d_r]
```

随后调用 `npu_fused_infer_attention_score`：

| 输入                         | Shape/含义                                             |
| ---------------------------- | ------------------------------------------------------ |
| `query`                    | 完整`Q_nope`，Head 维为 `d_n`                      |
| `key`                      | 上投影后的`K_nope`                                   |
| `value`                    | 上投影后的完整 V                                       |
| `query_rope`、`key_rope` | Q/K 的`d_r` 位置分量                                 |
| `atten_mask`               | Prefill 因果 Mask                                      |
| `actual_seq_lengths*`      | Packed Batch 的有效 Q/KV 长度                          |
| `scale`                    | `1/sqrt(d_n+d_r)`，包含 YaRN 修正时再乘 `mscale^2` |

算子输出是逐 Head 的 Attention 结果，逻辑 Shape 为 `[T,N_local,d_v]`。模型再将其展平并通过 `o_proj`，得到 `[T,H]`。这条路径的 Attention 维度较小，但需要展开逐 Head K/V，因此适合计算量占主导的稠密 Prefill。

#### 4. R1 Absorb Attention：在 latent 空间计算

R1 的 `forward_page_attention_absorb()` 和 `forward_page_attention_mla_prolog()` 使用 `npu_fused_infer_attention_score_v2`：

| 输入                      | Shape/含义                               |
| ------------------------- | ---------------------------------------- |
| `q_nope`                | `[T,N_local,d_c]`，已吸收 `W_UK`     |
| `k_nope`                | latent`nope_cache`，逻辑 Head 数为 1   |
| `value`                 | 与 Key 相同，仍使用 latent`nope_cache` |
| `query_rope`            | `[T,N_local,d_r]`                      |
| `key_rope`              | `rope_cache`，逻辑 Head 数为 1         |
| `block_table`           | 分页 KVCache 的读取映射                  |
| `actual_seq_qlen/kvlen` | 每个请求的有效 Q/KV 长度                 |
| `softmax_scale`         | Attention 缩放系数                       |

这里 `key=value=k_nope` 并不表示数学上的 K 与 V 权重相同，而是因为两次矩阵吸收后，Attention 得分计算与 Value 加权求和都可以围绕同一份 latent KV 完成。

FA v2 输出 latent Attention 结果 `[T,N_local,d_c]`。随后 MLA Epilog 执行：

```text
latent output x W_UV -> [T,N_local,d_v]
reshape/通信         -> [T,N_local*d_v]
o_proj               -> [T,H]
```

R1 代码使用 `npu_transpose_batchmatmul` 完成 `W_UV` 升维，然后执行 `o_proj` 和必要的 TP AllReduce。

对于单 Token Decode，代码使用 `sparse_mode=0` 且不传 Mask；MTP 使 Query 长度大于 1 时改用 `sparse_mode=3` 和因果 Mask。该差异用于保证多 Query 场景下的因果关系。

#### 5. V3.2-Exp Lightning Indexer：生成稀疏索引

V3.2-Exp 在 MLA Prolog 与 SFA 之间增加 Indexer Prolog 和 `npu_lightning_indexer`。Indexer Prolog 的主要计算为：

```text
qr -> wq_b -> Indexer Query [T,64,128]
x  -> wk   -> LayerNorm/RoPE -> Indexer Key [T,1,128]
x  -> weights_proj          -> Head 权重 [T,64]
Indexer Key -> 写入 indexer_key_cache
```

`npu_lightning_indexer` 的关键输入输出如下：

| 输入                             | 典型 Shape/含义                                                            |
| -------------------------------- | -------------------------------------------------------------------------- |
| `query`                        | `[T,64,128]`，Indexer Query                                              |
| `key`                          | 分页`indexer_key_cache`，逻辑 Shape 为 `[num_blocks,block_size,1,128]` |
| `weights`                      | `[T,64]`，用于聚合各 Indexer Head 的得分                                 |
| `block_table`                  | Indexer Key Cache 的分页读取映射                                           |
| `actual_seq_lengths_query/key` | 每个请求的有效 Query/Key 长度                                              |
| `sparse_count`                 | `2048`                                                                   |
| `sparse_mode`                  | `3`，遵循因果可见范围                                                    |

输出 `topk_indices` 的逻辑 Shape 为 `[T,1,2048]`，表示每个 Query 选中的历史 Token 位置。该输出不是 Attention Score，也不包含 KV 数据，只是后续 SFA 的稀疏读取索引。

#### 6. V3.2-Exp Sparse Flash Attention：只计算 TopK KV

`npu_sparse_flash_attention` 接收 Absorb Query、latent KVCache 和 LI 输出的索引：

| 输入                            | Shape/含义                          |
| ------------------------------- | ----------------------------------- |
| `query`                       | `q_nope [T,N_local,d_c]`          |
| `key`、`value`              | 同一份 latent`nope_cache`         |
| `query_rope`、`key_rope`    | Q/K 的`d_r` 位置分量              |
| `sparse_indices`              | `[T,1,2048]`，LI 输出的 TopK 位置 |
| `block_table`                 | 完整 MLA KVCache 的分页读取映射     |
| `actual_seq_lengths_query/kv` | 每个请求的有效长度                  |
| `scale_value`                 | Attention 缩放系数                  |
| `layout_query/layout_kv`      | `TND` / `PA_BSND`               |

算子内部先按 `sparse_indices` Gather KV，再完成 QK、Softmax 和 latent V 加权求和。Python 路径取其主输出并转置为 `[N_local,T,d_c]`，交给 MLA Epilog。C8 KVCache 场景使用 `npu_kv_quant_sparse_flash_attention`，输入还包含量化模式和 Scale，但数学输出含义相同。

启用 KVCache Offload 时，`npu_gather_selection_kv_cache` 先根据 `topk_indices` 从 Host 侧完整 KVCache 选择所需 KV，并复用设备侧已命中的块。其主要输出是当前步实际选入设备缓存的 KV 长度；SFA 随后使用局部选择缓存及重新生成的稀疏索引完成计算。

#### 7. MLA Epilog：恢复 Value 维度并输出 Hidden States

V3.2-Exp 的 `mla_epilog(absorb=True)` 与 R1 Absorb 尾部语义一致：

| 输入              | Shape/含义                                    |
| ----------------- | --------------------------------------------- |
| `attn_output`   | `[N_local,T,d_c]`，SFA 返回的 latent 加权和 |
| `kv_b_proj_w_v` | `[N_local,d_c,d_v]` 等价布局，即 `W_UV`   |
| `o_proj` 权重   | `[H,N*d_v]` 等价布局                        |

处理过程为：

1. `attn_output x W_UV`：将 latent 维度 `d_c=512` 恢复为每 Head 的 `d_v=128`。
2. 按 `o_proj_tp_size` 执行必要的 All-to-All，使 Token 和 Head 分片满足 `o_proj` 输入布局。
3. 执行 `o_proj`，并通过 ReduceScatter/AllReduce 合并并行分片。

最终输出为 `[T,H]`，可直接与 Decoder Layer 的残差分支相加。

#### 算子链路对比

| 模型路径                | Prolog/缓存                                           | Attention                            | Epilog            | 最终输出  |
| ----------------------- | ----------------------------------------------------- | ------------------------------------ | ----------------- | --------- |
| R1 Naive Prefill        | KV RMSNorm、RoPE、写 latent Cache，再展开完整 K/V     | Dense FA，K/V 为逐 Head 完整表示     | `o_proj`        | `[T,H]` |
| R1 Absorb Decode        | MLA Prolog 或拆分算子生成 Absorb Q，并写 latent Cache | FA v2，`key=value=latent KV`       | `W_UV + o_proj` | `[T,H]` |
| V3.2-Exp Prefill/Decode | MLA Prolog，同时生成 LI 可复用的`qr`                | LI 生成 TopK，SFA 读取稀疏 latent KV | `W_UV + o_proj` | `[T,H]` |

从接口角度看，MLA 算子链路遵循一个稳定契约：Prolog 将 `[T,H]` 转为 Absorb Query 并更新压缩 KVCache；Attention 在 `d_c` latent 空间完成得分计算与加权求和；Epilog 再通过 `W_UV` 和 `o_proj` 恢复为 `[T,H]`。R1 与 V3.2-Exp 的主要区别在 Attention 中间阶段：前者使用稠密 FA，后者先由 LI 选出 TopK，再使用 SFA。
