# DeepSeek-V4 Offline 推理完整流程

本文以以下命令为例，介绍从启动脚本、读取输入、执行 Prefill/Decode，到获取最终文本输出的完整流程。

```bash
bash executor/scripts/infer.sh \
  --model deepseek_v4 \
  --mode offline \
  --yaml ci_a3/deepseek_v4_flash_rank_16_16ep_w8a8_platform.yaml
```

## 1. 首先理解这条命令

该命令包含三个关键参数：

| 参数 | 含义 |
| --- | --- |
| `--model deepseek_v4` | 使用 `models/deepseek_v4/` 下的模型实现 |
| `--mode offline` | Prefill 和 Decode 在同一套 Offline 调度流程中执行，不启动 HTTP 服务 |
| `--yaml ...` | 配置文件相对于 `models/deepseek_v4/config/` 的路径 |

因此，实际使用的配置文件为：

```text
models/deepseek_v4/config/ci_a3/deepseek_v4_flash_rank_16_16ep_w8a8_platform.yaml
```

Offline 模式不是交互式服务。脚本启动后会自动读取 YAML 指定的数据集，批量生成文本，将结果写入日志，然后进程退出。

## 2. 运行前检查

### 2.1 检查 NPU 和软件环境

```bash
npu-smi info
python3 -c "import torch; import torch_npu; print(torch.__version__); print(torch_npu.__version__)"
```

推荐环境为 CANN 9.0.0、PyTorch 2.8.0、torch_npu 2.8.0 和 transformers 5.0.0。实际版本还应符合 DeepSeek-V4 README 与 `requirements.txt` 的要求。

### 2.2 配置 CANN 和节点 IP

编辑 `executor/scripts/set_env.sh`：

```bash
export IPs=('节点1_IP' '节点2_IP')

cann_path="/usr/local/Ascend/ascend-toolkit/latest"
source "$cann_path/bin/setenv.bash"
export ASCEND_HOME_PATH="$cann_path"
```

单机运行时，启动脚本会将 Offline IP 自动设置为本机 IP。多机运行时，必须按相同顺序在每台服务器上配置全部节点 IP，并在所有节点执行相同的启动命令。

还需检查 `executor/scripts/function.sh` 中的网卡前缀：

```bash
export HCCL_SOCKET_IFNAME=enp
```

如果实际网卡为 `eth0`，这里应配置为 `eth`。可以使用以下命令查看网卡：

```bash
ip addr
```

> 注意：当前 `executor/scripts/set_env.sh` 包含 `rm -rf /root/atc_data/`。执行推理脚本时会触发该删除操作，请先确认该目录允许被清理。

### 2.3 检查 YAML

当前 YAML 的关键配置为：

```yaml
model_config:
  model_name: "deepseek_v4"
  model_path: "/data/models/deepseek_v4_int8_w8a8"
  platform_version: "A3"
  next_n: 1
  exe_mode: "npugraph_ex"

data_config:
  dataset: "InfiniteBench"
  input_truncated_len: 8192
  temperature: 1.0

parallel_config:
  world_size: 16
  attn_tp_size: 1

scheduler_config:
  max_new_tokens: 256
  max_prefill_tokens: 8192
  batch_size: 256
```

必须确认：

1. `model_path` 指向真实存在且与量化配置匹配的权重目录。
2. `platform_version` 与机器一致。
3. `world_size=16` 与可用 NPU Rank 数一致。
4. `batch_size` 能被 Attention DP 数整除。
5. `input_truncated_len` 和 `max_prefill_tokens` 足够容纳输入。
6. `max_new_tokens=256` 表示每个请求最多保留 256 个有效生成 Token。

### 2.4 检查输入数据

当前 YAML 使用 `InfiniteBench`。程序优先读取：

```text
dataset/InfiniteBench/
```

如果本地目录不存在，代码会尝试按数据集加载逻辑获取数据。在无网络环境中，应提前将 InfiniteBench 数据放入该目录。

初次验证建议先使用仓库自带的简单 Prompt。将 YAML 改为：

```yaml
data_config:
  dataset: "default"
  input_truncated_len: 8192
  temperature: 0.0
```

输入文本位于：

```text
dataset/default_prompt.json
```

其格式为：

```json
{
  "text": "请输入希望模型续写或回答的内容"
}
```

`temperature: 0.0` 使用贪心选择，更适合首次验证；当前示例中的 `temperature: 1.0` 会进行随机采样，即使 Prompt 相同，输出也可能不同。

## 3. 启动 Offline 推理

在仓库根目录执行：

```bash
cd /Users/l/L/Code/AIProject/cann-recipes-infer

bash executor/scripts/infer.sh \
  --model deepseek_v4 \
  --mode offline \
  --yaml ci_a3/deepseek_v4_flash_rank_16_16ep_w8a8_platform.yaml
```

建议先确认参数解析结果：

```bash
bash executor/scripts/infer.sh --help
```

启动脚本会依次执行：

1. 加载 `executor/scripts/set_env.sh`。
2. 加载模型自己的 `models/deepseek_v4/set_env.sh`（如果存在）。
3. 将 YAML 解析为完整路径。
4. 校验模型目录、运行模式和 YAML 是否存在。
5. 从 YAML 读取 `world_size` 和硬件平台。
6. 初始化 HCCL 环境变量和结果目录。
7. 为每个本地 NPU Rank 启动一个 Python 进程。
8. 等待所有 Rank 推理结束。

命令返回 Shell 提示符后，说明启动脚本中的所有本地 Rank 已经退出。此时再检查日志，能够避免读取到尚未写完的结果。

## 4. Python 推理入口

Offline 模式首先查找：

```text
models/deepseek_v4/infer.py
```

如果该文件不存在，则使用统一入口：

```text
executor/offline/infer.py
```

DeepSeek-V4 当前进入统一 Offline 入口。每个 Rank 实际执行的命令等价于：

```bash
python3 executor/offline/infer.py \
  --yaml_file_path=models/deepseek_v4/config/ci_a3/deepseek_v4_flash_rank_16_16ep_w8a8_platform.yaml
```

启动脚本还会为每个进程设置：

```text
LOCAL_RANK       当前服务器上的 NPU 编号
RANK_ID          全局 Rank 编号
RANK_OFFSET      当前服务器的全局 Rank 起始偏移
WORLD_SIZE       总 Rank 数
WORK_DIR         models/deepseek_v4
RES_PATH         当前运行的结果目录
```

## 5. YAML 如何变成运行配置

`executor/offline/infer.py::main()` 读取 YAML，并创建 `InferenceConfig`。配置主要分为四部分：

| 配置段 | 作用 |
| --- | --- |
| `model_config` | 模型名、权重路径、精度、图模式、MTP 和优化开关 |
| `data_config` | 数据集、输入截断长度和采样参数 |
| `parallel_config` | TP、EP、DP、CP 等并行策略 |
| `scheduler_config` | Batch、KV Block、Prefill 上限和生成长度 |

当前配置中：

```text
world_size = 16
attn_tp_size = 1
attn_dp_size = world_size / attn_tp_size = 16
batch_size_per_dp_rank = batch_size / attn_dp_size = 256 / 16 = 16
```

因此，每个 Attention DP Rank 处理 16 个请求。该单机 16 Rank 配置的 256 个请求会分散到 `log_0.log` 至 `log_15.log` 中，而不是全部出现在 Rank 0 的终端输出中。

## 6. 输入如何进入模型

### 6.1 读取 Prompt

`executor/offline/infer.py::generate_prompt()` 根据 `data_config.dataset` 读取输入：

```text
default       -> dataset/default_prompt.json
LongBench     -> dataset/LongBench/ 或远端数据集
InfiniteBench -> dataset/InfiniteBench/ 或对应数据加载逻辑
```

对于 LongBench 和 InfiniteBench，程序还会通过 `build_dataset_input()` 添加摘要任务的 Prefix/Suffix，并按照 `input_truncated_len` 截断原文。

### 6.2 按 DP Rank 分配 Prompt

当 `attn_dp_size > 1` 时，每个 Rank 根据其 DP 编号获得不同的 Prompt 切片。当前配置中，每个 DP Rank 获得 16 条输入。

如果一个 Attention TP Group 包含多个 Rank，同一 TP Group 内的 Rank 会协同处理同一批 Prompt；完整输出应按 DP Group 汇总，不能简单把所有 TP Rank 日志拼接，否则会出现重复结果。当前 YAML 的 `attn_tp_size=1`，不存在这种 TP 重复。

### 6.3 Chat Template 和 Tokenizer

`OfflineInference.generate()` 会将普通字符串转换为 Chat Message：

```python
[{"role": "user", "content": prompt}]
```

Scheduler 随后执行：

```text
Chat Message
  -> tokenizer.apply_chat_template(add_generation_prompt=True)
  -> tokenizer(..., truncation=True, max_length=input_truncated_len)
  -> input_ids
```

DeepSeek-V4 使用仓库注册的 `DeepseekV4Tokenizer`。Tokenize 后的 `input_ids` 保存在 `Request` 中，等待 Scheduler 组 Batch。

如果 Token 数超过 `max_prefill_tokens`，请求会以 `prompt_too_long` 结束，不进入模型计算。

## 7. 模型和 KVCache 如何初始化

`OfflineInference` 创建 `ExecutionEngine`，随后完成：

1. 根据 `deepseek_v4` 从 `executor/core/support_models.py` 获取模型类和配置类。
2. 从 `model_path` 加载模型配置、Tokenizer 和权重。
3. 初始化 HCCL 通信组及 TP/EP/DP 等子通信组。
4. 读取模型的 `get_cache_info()`。
5. 为各层分配分页 KVCache 和其他模型缓存。
6. 如果 `next_n > 0`，加载并初始化 MTP 模型。
7. 执行一次 Prefill 和一次 Decode Warm-up。
8. `npugraph_ex` 模式下在 Warm-up 中完成图编译或图捕获。

Warm-up 使用随机 Dummy Token，不是用户输入，其结果不会作为最终输出。

## 8. Prefill 阶段

Scheduler 首先从等待队列选取请求，并受以下条件限制：

- `max_prefill_tokens`
- `batch_size_per_dp_rank`
- KVCache 可用 Block 数
- CP Mini Batch 配置

多个请求的 Token 会拼接成 Packed 输入：

```text
input_ids: [所有请求有效 Token 的总数]
seq_lens:  [每个请求的实际长度]
```

ExecutionEngine 再生成：

```text
position_ids
actual_seq_lengths_q / actual_seq_lengths_kv
slot_mapping
block_table
causal attention_mask
```

DeepSeek-V4 模型前向执行 Embedding、Decoder Layers、Attention/MoE 和 LM Head，返回每个请求最后一个有效 Prompt Token 对应的 Logits。

Sampler 根据 `temperature`、`top_k`、`top_p` 和 `seed` 选择第一个输出 Token，并将其追加到 `Request.output_id_list`。与此同时，Prefill 产生的 KV 状态被保留在分页 KVCache 中，供后续 Decode 使用。

## 9. Decode 阶段

Prefill 完成后，请求从等待队列进入 Decode 运行队列。每一轮 Decode 执行：

```text
上一步生成的 Token
  -> 构造 position_ids、kv_len、slot_mapping 和 block_table
  -> 模型 Decode Forward
  -> 生成下一 Token 的 Logits
  -> Sampler 选择 Token
  -> 追加到 output_id_list
  -> 更新 KVCache
```

当前 YAML 配置 `next_n=1`，会启用 MTP 推测解码。主模型验证 MTP 候选 Token，将接受的 Token 追加到结果中，因此一次 Decode Forward 可能确认多个有效 Token。

Offline 模式为了让所有 DP Rank 保持同步，即使某个请求提前遇到 EOS，也可能继续执行到统一的 Decode Step 上限；但程序会记录第一个 EOS 对应的 `valid_output_len`，最终只解码有效部分。因此日志中的最终文本不会包含 EOS 后的无效 Token。

请求在以下条件之一满足时得到结束原因：

- 遇到 EOS：`finish_reason=stop`
- 达到 `max_new_tokens`：`finish_reason=length`
- 输入过长或执行异常：对应错误原因

## 10. Token 如何转换为最终文本

调度循环结束后，`OfflineInference.generate()` 从每个 Request 取出有效 Token：

```python
valid_output_ids = request.output_id_list[:request.valid_output_len]
```

随后执行：

```python
output_text = tokenizer.decode(
    valid_output_ids,
    skip_special_tokens=True,
)
```

最终形成：

```text
GenerationOutput(
    prompt=原始输入,
    output_text=生成文本,
    finish_reason=结束原因,
)
```

统一入口的 `log_results()` 将 `output_text` 写入日志：

```text
Request 0: outputs: 模型生成的文本
Request 1: outputs: 模型生成的文本
...
```

它还会输出 Decode 平均耗时、MTP 接受率和等效时延等性能信息。

## 11. 在哪里查看输出

### 11.1 终端输出

启动脚本只对本机 `LOCAL_RANK=0` 使用 `tee`，因此终端实时显示的是 Rank 0 日志。其他 Rank 的输出只写入文件。

### 11.2 日志目录

结果目录格式为：

```text
models/deepseek_v4/res/<YYYYMMDD>/<model_name>_<yaml_name>/
```

本例在 2026 年 8 月 10 日运行时，对应：

```text
models/deepseek_v4/res/20260810/
  deepseek_v4_deepseek_v4_flash_rank_16_16ep_w8a8_platform/
    log_0.log
    log_1.log
    ...
    log_15.log
```

实际日期以运行当天为准。可以查找最新结果：

```bash
find models/deepseek_v4/res \
  -type f \
  -name 'log_*.log' \
  -path '*deepseek_v4_flash_rank_16_16ep_w8a8_platform*' \
  -print
```

### 11.3 查看 Rank 0 输出

```bash
RESULT_DIR="models/deepseek_v4/res/$(date +%Y%m%d)/deepseek_v4_deepseek_v4_flash_rank_16_16ep_w8a8_platform"

less "$RESULT_DIR/log_0.log"
```

只定位模型文本：

```bash
rg -n 'Request [0-9]+: outputs:' "$RESULT_DIR/log_0.log"
```

如果输出文本包含换行，应使用 `less` 阅读原始日志，不能只依赖 `rg` 返回的首行。

### 11.4 查看全部 16 个 Rank 的输出

当前配置的完整 256 条结果分散在 16 份日志中：

```bash
rg -n 'Request [0-9]+: outputs:' "$RESULT_DIR"/log_*.log
```

生成一个便于保存和检索的汇总日志：

```bash
ALL_OUTPUTS="$RESULT_DIR/all_outputs.log"
: > "$ALL_OUTPUTS"

for log_file in "$RESULT_DIR"/log_*.log; do
  printf '\n===== %s =====\n' "$(basename "$log_file")" >> "$ALL_OUTPUTS"
  sed -n '/Request [0-9][0-9]*: outputs:/,/decode average inference time cost/p' \
    "$log_file" >> "$ALL_OUTPUTS"
done

less "$ALL_OUTPUTS"
```

该命令保留每个 Rank 的分隔符和可能包含换行的生成文本。它是日志汇总，不是结构化 JSON。

多机运行时，每台服务器都使用自己的本地 `LOCAL_RANK` 文件名，因此需要从所有节点收集日志。不要将不同节点的 `log_0.log` 互相覆盖；汇总前应按节点名建立子目录。

## 12. 如何判断推理成功

首先确认命令退出码：

```bash
echo $?
```

值为 `0` 表示启动脚本正常结束，但还应检查所有 Rank 日志。

检查严重错误：

```bash
rg -n 'Traceback|RuntimeError|ERROR|Out of memory|HCCL.*error' \
  "$RESULT_DIR"/log_*.log
```

检查输出和性能字段：

```bash
rg -n 'Request [0-9]+: outputs:|average inference time|acceptance rate|accept rate' \
  "$RESULT_DIR"/log_*.log
```

完整成功条件为：

1. 所有预期 Rank 都生成了日志。
2. 日志中没有 Traceback、OOM 或 HCCL 错误。
3. 日志包含 `Request N: outputs:`。
4. 输出文本可读、没有明显无限重复或乱码。
5. 所有 Rank 正常退出。

## 13. 常见问题

### 13.1 终端只有部分结果

这是正常现象。终端只显示 Rank 0；应检查 `log_0.log` 至 `log_15.log`。

### 13.2 没有 `Request N: outputs:`

按以下顺序检查：

1. 是否仍在模型加载、权重加载或 Warm-up。
2. 是否发生图编译错误。
3. 数据集是否存在。
4. Prompt 是否超过 `max_prefill_tokens`。
5. 是否有某个 Rank 提前退出，导致其他 Rank 等待通信。

### 13.3 输出为空

检查日志中请求是否被标记为 `prompt_too_long` 或错误结束；再检查 Tokenizer、权重类型和量化配置是否匹配。空文本也可能来自生成结果只有特殊 Token，因 `skip_special_tokens=True` 被过滤。

### 13.4 输出每次不同

当前配置 `temperature: 1.0` 使用随机采样。首次验证可改为：

```yaml
data_config:
  temperature: 0.0
```

如需可复现的随机采样，还应在 YAML 的 `data_config` 中设置固定 `seed`。

### 13.5 启动后提示已有 infer.py 或 server.py

启动脚本会检查是否已有推理进程。先确认该进程是否仍属于正在运行的任务，不要直接终止未知进程：

```bash
pgrep -af 'python.*(infer|server)\.py'
```

### 13.6 OOM

优先降低：

1. `scheduler_config.batch_size`
2. `data_config.input_truncated_len`
3. `scheduler_config.max_new_tokens`
4. `scheduler_config.max_prefill_tokens`

修改 `batch_size` 时仍需满足其能被 `attn_dp_size` 整除。当前 `attn_dp_size=16`，因此 Batch Size 至少应按 16 的倍数调整。

## 14. 完整调用链

```text
executor/scripts/infer.sh
  -> 解析 --model/--mode/--yaml
  -> validate_infer_args.py 校验输入
  -> function.sh::launch("offline")
     -> get_rank() 读取 world_size 和节点数
     -> check_env_vars() 初始化 HCCL 与结果目录
     -> launch_infer_task() 启动每个本地 Rank
        -> executor/offline/infer.py::main()
           -> InferenceConfig.from_dict()
           -> generate_prompt() 读取数据集
           -> OfflineInference(config)
              -> ExecutionEngine
              -> 加载模型、Tokenizer、权重和 KVCache
              -> warm_up() 执行 Dummy Prefill/Decode
           -> OfflineInference.generate(prompts)
              -> Scheduler.add_request()
              -> Chat Template + Tokenizer
              -> Scheduler.run_step()
                 -> Prefill Batch
                 -> ExecutionEngine.forward_batch()
                 -> DeepSeek-V4 Forward + LM Head
                 -> Sampler 选择首个 Token
                 -> Decode Batch 循环
                 -> Token 追加到 Request.output_id_list
              -> tokenizer.decode(valid_output_ids)
              -> GenerationOutput.output_text
           -> log_results()
              -> Request N: outputs: ...
              -> 写入 log_<rank>.log
```

## 15. 最短操作清单

```bash
# 1. 进入仓库
cd /Users/l/L/Code/AIProject/cann-recipes-infer

# 2. 确认 NPU
npu-smi info

# 3. 确认 YAML 中的 model_path、world_size、dataset
sed -n '1,160p' \
  models/deepseek_v4/config/ci_a3/deepseek_v4_flash_rank_16_16ep_w8a8_platform.yaml

# 4. 启动 Offline 推理
bash executor/scripts/infer.sh \
  --model deepseek_v4 \
  --mode offline \
  --yaml ci_a3/deepseek_v4_flash_rank_16_16ep_w8a8_platform.yaml

# 5. 定位结果目录
RESULT_DIR="models/deepseek_v4/res/$(date +%Y%m%d)/deepseek_v4_deepseek_v4_flash_rank_16_16ep_w8a8_platform"
ls -lh "$RESULT_DIR"

# 6. 检查错误
rg -n 'Traceback|RuntimeError|ERROR|Out of memory|HCCL.*error' \
  "$RESULT_DIR"/log_*.log

# 7. 查看所有生成结果
rg -n 'Request [0-9]+: outputs:' "$RESULT_DIR"/log_*.log
```

当前实现的最终文本以日志形式输出，并不会自动生成独立 JSON 文件。若后续需要程序化消费结果，建议扩展 `executor/offline/infer.py::log_results()`，将 `GenerationOutput` 中的 `prompt`、`output_text` 和 `finish_reason` 写入 JSONL 文件。
