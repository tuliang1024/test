# MLA 常见问题解答

## 案例：LongCat-2.0 中 MLA Naive 与 Absorb 的选择

LongCat-2.0 在 Prefill 阶段采用 3S DSA。以 Batch Size 为 1、序列长度为 64K 的推理场景为例，Lightning Indexer 为每个 Query Token 选择 TopK=2048 个 KV Token。MLA 有以下三种实现方案。

### 方案一：MLA Naive + Sparse Mask

Prefill MLA 使用 Naive 模式。每个 Query Token 与所有历史 KV Token 计算 Attention，仅在 Softmax 前通过 Attention Mask 过滤 TopK 以外的 Token。两个批量矩阵乘的 Shape 如下。

| 计算 | Batch | M | K | N |
| --- | ---: | ---: | ---: | ---: |
| BMM1（Q x K^T） | 1 x 128 | 64K | 192 | 64K |
| BMM2（P x V） | 1 x 128 | 64K | 64K | 128 |

该方案的计算量与 Full Attention 相同，无法获得 DSA 的稀疏计算收益，不适合长序列场景。

### 方案二：MLA Naive + Sparse Attention

Prefill MLA 使用 Naive 模式，每个 Query Token 仅与 TopK=2048 个 KV Token 计算 Attention。由于每个 Query Token 独立选择 KV Token，序列长度 64K 被外提到 Batch 轴，BMM 的 M 轴为 1。

| 计算 | Batch | M | K | N |
| --- | ---: | ---: | ---: | ---: |
| BMM1（Q x K^T） | 1 x 64K x 128 | 1 | 192 | 2048 |
| BMM2（P x V） | 1 x 64K x 128 | 1 | 2048 | 128 |

该方案的 BMM 计算量约为方案一的 `2048 / 64K = 1/32`，但存在两个问题：

- BMM 的 M 轴为 1，矩阵乘计算效率较低。
- 每个 Query Token、每个 Attention Head 都需要独立读取选中的 KV，KV 数据难以跨 Query 和 Head 复用，容易形成 HBM 访存瓶颈。

### 方案三：MLA Absorb + Sparse Attention

Prefill MLA 使用 Absorb 模式，与 Decode 保持一致。每个 Query Token 同样只与 TopK=2048 个 KV Token 计算 Attention，但 128 个 Query Head 共享同一份 latent KV。

| 计算 | Batch | M | K | N |
| --- | ---: | ---: | ---: | ---: |
| BMM1（Q x K^T） | 1 x 64K | 128 | 576 | 2048 |
| BMM2（P x V） | 1 x 64K | 128 | 2048 | 512 |

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

因此，按两个 BMM 的累计 HBM 读取量计算，方案三约降至方案二的 `1/37.6`。若比较 BMM 使用的逐 Head 展开 K/V 与 latent KV 表示，元素数之比为：

```text
[128 x (192 + 128)] / 576 = 71.1
```

上述 `71.1` 倍要求 RoPE Key 也按 Head 展开；若实现单独共享 RoPE Key，比例应按实际 Cache Layout 重新计算。实际收益还受算子融合、片上缓存复用、数据类型、对齐方式和 Gather 访问效率影响，因此通常表述为 KV HBM 访存量降低数十倍。

综合计算量、HBM 访存量和矩阵乘效率，LongCat-2.0 选择方案三完成 Prefill 部署，并统一 Prefill 与 Decode 的 MLA 计算路径。

## FAQ 遗留问题

### 1. 什么时候使用矩阵吸收？什么时候使用非吸收？

MLA 矩阵吸收与非吸收在数学上等价，差异在于 KV 上投影矩阵位于 Attention 的哪一侧执行。

设 KV latent 为 `C`，Key 和 Value 上投影矩阵分别为 `W_UK` 和 `W_UV`。非吸收路径为：

```text
K = C x W_UK
V = C x W_UV
P = Softmax(Q x K^T)
O = P x V
```

利用矩阵乘法结合律，吸收路径可改写为：

```text
Q_latent = Q x W_UK^T
P = Softmax(Q_latent x C^T)
O = (P x C) x W_UV
```

吸收模式将 `W_UK` 从历史 KV 侧移动到当前 Query 侧，将 `W_UV` 从历史 Value 侧移动到 Attention 输出侧，使 Attention 直接在 latent 空间完成。

#### 计算量对比

定义：

| 符号 | 含义 |
| --- | --- |
| `H` | Attention Head 数 |
| `Q` | Query Token 数 |
| `S` | KV 序列长度或每个 Query 实际选择的 KV 数 |
| `d_c` | KV latent 维度 |
| `d_k` | NoPE Key 维度 |
| `d_r` | RoPE 维度 |
| `d_v` | Value 维度 |

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

| 模式 | BMM Shape 特征 | 算术强度 | 常见瓶颈 |
| --- | --- | ---: | --- |
| Naive Sparse Attention | `M=1`，KV 按 Query、Head 独立 Gather | 低 | HBM 带宽、离散访存和矩阵乘利用率 |
| Absorb Sparse Attention | `M=H`，各 Head 共享 latent KV | 高 | 可能从带宽受限转为计算受限 |

因此，不能仅根据 FLOPs 判断性能：

- 若 Naive 位于 Roofline 的带宽受限区，减少 FLOPs 对耗时帮助有限；Absorb 即使增加计算量，也可能因显著减少 HBM 流量而更快。
- 若 Absorb 仍位于带宽受限区，其理想收益主要由 HBM 搬运量下降决定；实际收益还取决于 Gather 的有效带宽和片上缓存复用。
- 若 Absorb 越过 Roofline 拐点进入计算受限区，继续减少 KV 搬运的收益会减弱，此时应重点优化 BMM 利用率和额外投影计算。
- 若 Naive 已具有较高的数据复用并进入计算受限区，Absorb 增加的 latent 维度和投影计算可能直接转化为额外耗时。

对应到推理场景：

| 场景 | Roofline 特征 | 推荐模式 |
| --- | --- | --- |
| 单 Token Decode、长上下文 | Query 数少，历史 KV 读取量大，Naive AI 低 | Absorb |
| 稀疏 Prefill，KV 按 Query 独立选择 | Query 间难以复用 KV，Naive 的 `M=1` 且离散访存 | Absorb |
| 稠密 Prefill | 同一 KV 可被多个 Query 复用，Naive AI 随 Query Block 增大 | 优先评估 Naive |
| 短序列或 KV 可驻留片上缓存 | HBM 搬运占比下降，额外 FLOPs 更敏感 | 以实测结果为准 |
| KV 量化 | 单元素字节数下降，两种模式的 AI 均提高 | 重新按量化数据类型计算并实测 |

实际选型时，应从 Profiling 数据获得 `FLOPs`、`HBM Bytes`、有效带宽和计算利用率，再与目标硬件对应数据类型的 `AI_ridge` 比较。稀疏 Gather 通常无法达到连续读取的峰值带宽，因此应使用实测 `BW_eff`，不能直接代入硬件标称带宽。

#### 选型建议

| 场景 | 推荐模式 | 主要原因 |
| --- | --- | --- |
| 单 Token Decode、长上下文或大 Batch Decode | Absorb | 避免展开历史 K/V，显著降低 KV Cache 读取量 |
| MTP Decode，且 Query 数远小于 KV 长度 | Absorb | 多个 Query 仍可复用 latent KV，访存收益通常占优 |
| 稀疏 Prefill，TopK KV 按 Query 独立选择 | Absorb | 可跨 Head 共享 latent KV，并将 Head 维放入 BMM 的 M 轴 |
| 稠密长序列 Prefill | 优先评估 Naive | Naive 的 Attention 维度更小，理论计算量更低 |
| 短序列、低 Batch | 以实测结果为准 | KV 搬运收益有限，额外投影和算子调度可能抵消收益 |
| Absorb 算子不支持目标维度、布局或量化模式 | Naive | 首先保证算子兼容性和精度正确性 |
| 需要统一 Prefill 与 Decode 路径 | 优先评估 Absorb | 可降低工程复杂度，但必须验证 Prefill 计算开销 |

总体原则如下：

- Decode 更关注历史 KV 的容量和 HBM 搬运，通常选择 Absorb。
- 稠密 Prefill 更关注 Attention 的计算量，通常优先评估 Naive。
- 稀疏 Prefill 需要同时考虑 BMM Shape 与 KV Gather 访存；当 Naive 路径的 M 轴过小且 KV 无法跨 Head 复用时，Absorb 可能更快。
- 最终选择应以目标序列长度、Batch Size、数据类型和实际 Profiling 结果为准，不能仅依据 FLOPs 判断。
