## Deepseek v4模型结构介绍

### Deepseek v4 Flash和Pro模型

Deepseek v4 系列主要包含两个核心版本：**DeepSeek-V4-Pro**和**DeepSeek-V4-Flash**。DeepSeek‑V4‑Pro模型，总参数量为1.6T，每Token激活参数量为49B；DeepSeek‑V4‑Flash模型，总参数量为284B，每Token激活参数量为13B。模型均支持1M长度上下文。

Flash 和 Pro 的模型结构基本一致，区别主要不在模型架构，而在参数规模、容量配置以及长距离建模侧重点。

两者的主要区别可以概括为：

- **Pro 更深**：`num_hidden_layers` 从 Flash 的 `43` 提升到 Pro 的 `61`
- **Pro 更宽**：`hidden_size` 从 `4096` 提升到 `7168`
- **Pro 的 attention 容量更大**：`num_attention_heads` 从 `64` 提升到 `128`
- **Pro 的 routed experts 更多**：`n_routed_experts` 从 `256` 提升到 `384`
- **Pro 的 expert 中间容量更大**：`moe_intermediate_size` 从 `2048` 提升到 `3072`
- **Pro 的远程压缩检索更激进**：`index_topk` 从 `512` 提升到 `1024`，整体更偏向长距离全局建模；而 Flash 更强调效率与均衡

| 参数                        | Flash     | Pro       | 说明                                       |
| :-------------------------- | :-------- | :-------- | :----------------------------------------- |
| `hidden_size`             | 4096      | 7168      | token hidden 维度                          |
| `num_hidden_layers`       | 43        | 61        | 主干 Block 层数                            |
| `num_attention_heads`     | 64        | 128       | Q 头数                                     |
| `o_groups`                | 8         | 16        | O 投影分组数                               |
| `n_routed_experts`        | 256       | 384       | routed experts 数量                        |
| `moe_intermediate_size`   | 2048      | 3072      | expert 中间维度                            |
| `index_topk`              | 512       | 1024      | CSA 中远程 compressed KV 的 top-k 检索上限 |
| `max_position_embeddings` | 1,048,576 | 1,048,576 |                                            |

### Deepseek v4模型结构

#### DeepSeek-V4 系列的总体架构

deepseek v4系列仍然采用Transformer 架构和MTP模块，主要的变化：

1. 引入Manifold-Constrained Hyper-Connections（mHC）
2. 一种混合注意力架构，结合了Compressed Sparse Attention（CSA）和Heavily Compressed Attention（HCA）
3. 采用 Muon 作为优化器
4. Moe细微调整（相比dsv3），将最初的几个FFN层替换为Hash Moe层

![1784657677368](image/Deepseekv4模型结构/1784657677368.png)

> Deepseek v4模型结构图：[github.com/CalvinXKY/InfraTech/blob/main/models/deepseek_v4/deepseek_v4_architecture.jpg](https://github.com/CalvinXKY/InfraTech/blob/main/models/deepseek_v4/deepseek_v4_architecture.jpg)

#### mHC

![1784657956080](image/Deepseekv4模型结构/1784657956080.png)

mHC是对于传统残差连接的扩展，它将hidden_state从一路扩展到多路，在Attention/MoE计算前通过Pre Mapping融合回一路，保持Attention/MoE的计算过程不变。对于Attention和MoE的输出结果，再通过Post Mapping扩展回多路，同时多路残差通过Res Mapping进行特征融合，融合结果与Post Mapping结果加和，得到mHC输出。

残差网络：$x_{l+1}=x_l+F_l(x_l)$

HC：$x_{l+1}=\mathcal{H}^{res}_{l}x_l+\mathcal{H}^{post}_{l}\mathcal{F}(\mathcal{H}^{pre}_{l}x_l,\mathcal{W}_l)$

mHC论文中提出的mHC结构，在原始HC的基础上将残差矩阵  $\mathcal{H}^{pre}_{l}$、$\mathcal{H}^{post}_{l}$和 $\mathcal{H}^{res}_{l}$做一个约束，保证在训练过程中该矩阵权重不会出现爆炸的值（比如远大于1）或者弥散的值（比如无限接近于0）。

#### Hybrid Attention with CSA and HCA

##### Compressed Sparse Attention

![1784659155481](image/Deepseekv4模型结构/1784659155481.png)

CSA（Compressed Sparse Attention）：压缩率 $m=4$，每 4 个 token 的 KV 压成 1 个 entry，再在压缩后的 entries 上跑 DeepSeek Sparse Attention（DSA★）——每个 query 通过 lightning indexer 选 top-k 个压缩 entry 做 attention（V4-Pro 取 $k=1024$）。压缩过程用两条独立的 KV 序列 $\mathcal{C}^{a}$、$\mathcal{C}^{b}$ 配合 softmax 归一化的门控权重做加权合并，相邻两个压缩块共享 $\mathcal{C}^{a}$ 与 $\mathcal{C}^{b}$ 的部分索引，形成重叠压缩。

##### Heavily Compressed Attention

![1784659742037](image/Deepseekv4模型结构/1784659742037.png)

HCA（Heavily Compressed Attention）：压缩率 $m'=128$，每 128 个 token 的 KV 压成 1 个 entry

##### Compressor

DeepSeek-V4 采用了一个新的attention架构，Compress-4-Attention (C4A) 和Compress-128-Attention (C128A). 具体来说是将每 4 或 128 个 token 的$KV$cache 压缩成一个，然后每个token与这些压缩的$KV$cache去进行 Attention 计算。在长序列的情况下，C4A 和 C128A 可以有效地减少计算开销。

![1784660640036](image/Deepseekv4模型结构/1784660640036.png)


**C4A层**

C4A层的 Compressor 包含如下一系列计算操作，给定输入$X\in \mathbb{R}^{s\times h}$，其中 $s$是 序列长度，$h$ 是hidden size，首先计算其对应的 $2$ 个$KV$输入 $C^a, C^b \in \mathbb{R}^{s \times d}$,与压缩权重 $Z^a, Z^b \in \mathbb{R}^{s \times d}$，其中 $d$ 是 head dimension. 具体公式如下：

$$
\begin{aligned}
C^a&=X\cdot W^{aKV}, C^b=X\cdot W^{bKV}, \\
Z^a&=X\cdot W^{aGate}, Z^b=X\cdot W^{bGate},
\end{aligned}
$$

其中$W^{aKV}, W^{bKV}, W^{aGate}, W^{bGate} \in \mathbb{R}^{h \times d}$是C4A对应KV和压缩权重的权重参数。

长度为$s$的KV序列$C^a, C^b$中的每4个KV会被压缩成1个
$C^{\text{Comp}} \in \mathbb{R}^{\frac{s}{4} \times d}$，其第 $i$ 行 $C_{i}^{\text{Comp}} \in \mathbb{R}^{1 \times d}$ 的计算公式如下：

$$
\begin{aligned}
C_{i}^{\text{Comp}}
&= \frac{\sum_{j=4i}^{4(i+1)-1}{e^{Z_j^a+B_{j-4i}}} \odot C^a_j + \sum_{j=4(i+1)}^{4(i+2)-1}{e^{Z_j^b+B_{j-4i}}} \odot C^b_j}{\sum_{j=4i}^{4(i+1)-1}{e^{Z_j^a+B_{j-4i}}} + \sum_{j=4(i+1)}^{4(i+2)-1}{e^{Z_j^b+B_{j-4i}}}} \\
&= \left[1\right]_{1\times8} @ (softmax(\left[Z^a_{\left[4(i-1)+1:4i,:\right]} ; Z^b_{\left[4i+1:4(i+1),:\right]}\right] + B) \odot \left[C^a_{\left[4(i-1)+1:4i,:\right]} ; C^b_{\left[4i+1:4(i+1),:\right]}\right])
\end{aligned}
$$

其中 $B \in \mathbb{R}^{8 \times d}$ 为 $C^a, C^b$ 对应的positional biases。

**C4A层**

C4A层的 Compressor 包含如下一系列计算操作，给定输入$X\in \mathbb{R}^{s\times h}$，其中 $s$是 序列长度，$h$ 是hidden size，首先计算其对应的 $2$ 个$KV$输入 $C^a, C^b \in \mathbb{R}^{s \times d}$,与压缩权重 $Z^a, Z^b \in \mathbb{R}^{s \times d}$，其中 $d$ 是 head dimension. 具体公式如下：

$$
\begin{aligned}
C^a&=X\cdot W^{aKV}, C^b=X\cdot W^{bKV}, \\
Z^a&=X\cdot W^{aGate}, Z^b=X\cdot W^{bGate},
\end{aligned}
$$

其中$W^{aKV}, W^{bKV}, W^{aGate}, W^{bGate} \in \mathbb{R}^{h \times d}$是C4A对应KV和压缩权重的权重参数。

长度为$s$的KV序列$C^a, C^b$中的每4个KV会被压缩成1个
$C^{\text{Comp}} \in \mathbb{R}^{\frac{s}{4} \times d}$，其第 $i$ 行 $C_{i}^{\text{Comp}} \in \mathbb{R}^{1 \times d}$ 的计算公式如下：

$$
\begin{aligned}
C_{i}^{\text{Comp}}
&= \frac{\sum_{j=4i}^{4(i+1)-1}{e^{Z_j^a+B_{j-4i}}} \odot C^a_j + \sum_{j=4(i+1)}^{4(i+2)-1}{e^{Z_j^b+B_{j-4i}}} \odot C^b_j}{\sum_{j=4i}^{4(i+1)-1}{e^{Z_j^a+B_{j-4i}}} + \sum_{j=4(i+1)}^{4(i+2)-1}{e^{Z_j^b+B_{j-4i}}}} \\
&= \left[1\right]_{1\times8} @ (softmax(\left[Z^a_{\left[4(i-1)+1:4i,:\right]} ; Z^b_{\left[4i+1:4(i+1),:\right]}\right] + B) \odot \left[C^a_{\left[4(i-1)+1:4i,:\right]} ; C^b_{\left[4i+1:4(i+1),:\right]}\right])
\end{aligned}
$$

其中 $B \in \mathbb{R}^{8 \times d}$ 为 $C^a, C^b$ 对应的positional biases。

**C128A层**
相比C4A, C128A层的 Compressor 会以更大的压缩比率对KV序列去进行压缩，并且只依赖单一KV序列$C \in \mathbb{R}^{s \times d}$和压缩权重$Z \in \mathbb{R}^{s \times d}$，其中

$$
\begin{aligned}
C&=X\cdot W^{KV}, \\
Z&=X\cdot W^{Gate}
\end{aligned}
$$

$W^{KV}, W^{Z}$ 是C128A对应KV和压缩权重的权重参数。

然后长度为$s$的KV序列$C$中的每 $128$ 个$KV$会被压缩成 $1$ 个 $C^{\text{Comp}} \in \mathbb{R}^{\frac{s}{128} \times d}$，其第 $i$ 行

$$
\begin{aligned}
C_{i}^{\text{Comp}}
&= \frac{\sum_{j=128i}^{128(i+1)-1}{e^{Z_j+B_{j-128i}}} \odot C_j}{\sum_{j=128i}^{128(i+1)-1}{e^{Z_j+B_{j-128i}}}} \\
&= \left[1\right]_{1\times128} @ (softmax(Z_{\left[128(i-1)+1:128i,:\right]} + B) \odot C_{\left[128(i-1)+1:128i,:\right]})
\end{aligned}
$$

$i = 1, \cdots, \frac{s}{128},$ 其中 $B \in \mathbb{R}^{128 \times d}$ 为 $C$ 对应的positional biases.


#### Moe

![1784660805519](image/Deepseekv4模型结构/1784660805519.png)
