* 什么是superkernel

  可选）标定SuperKernel范围。

  使用如下语句块（super_kernel_scope_begin和super_kernel_scope_end），语句块内的算子将被融合为一个SuperKernel进行计算。

  ```python
  torch.npu.super_kernel_scope_begin(scope_name: str)
  待融合的算子操作
  torch.npu.super_kernel_scope_end(scope_name: str)
  ```

  scope_name：表示该范围内算子融合后的SuperKernel名称，相同的scope\_name代表相同的融合范围，由用户控制。若传入None，则该范围内的算子不进行SuperKernel融合。

  > [!NOTE] ✏️ 说明
  >

superkernel介绍：SuperKernel是一种算子二进制融合技术。与源码融合不同，它聚焦于内核函数（Kernel）的二进制调度方案优化，在已编译的二进制代码基础上融合创建一个超级Kernel函数（简称SuperKernel）

与单算子下发相比，SuperKernel技术能够优化任务调度的等待时间和调度开销，并可利用Task间隙资源进一步优化算子头开销。


![1784823465566](image/SuperKernel/1784823465566.png)

开启SuperKernel融合优化后，系统自动识别图内可被融合的算子，在SuperKernel内以子函数调用的方式依次执行。同时提供标定SuperKernel范围的能力，支持用户根据实际业务需求对融合范围内的算子进行标记和优化配置。



* superkernel优化注意事项
  * 可能有些算子是不支持融合的：SuperKernel融合会按照网络中算子的顺序依次判断是否可被融合。当**识别到不可融合的算子时**，系统会生成第一段SuperKernel，并自动跳过该算子进行第二段SuperKernel融合。（不支持融合算子问题定位指导文档[gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/cases/superkernel_cases.md](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/cases/superkernel_cases.md)）



* superkernel使用方式1
  * ge-graph的superkernel开启方式

用户自行分析模型脚本中可被融合的算子。

标定SuperKernel范围。

使用如下with语句块（super_kernel），语句块内算子均被融合为一个超级Kernel进行计算。



```
with torchair.scope.super_kernel(scope: str, options: str = ''):
```

scope：表示上下文算子被融合的SuperKernel名，相同的scope代表相同的范围，由用户控制。

options：表示融合SuperKernel的编译选项（具体选项可参考文档：[www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00035.html](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00035.html))


* npugraph-ex图模式的superkernel开启方法



1.（可选）标定SuperKernel范围。

使用如下语句块（super_kernel_scope_begin和super_kernel_scope_end），语句块内的算子将被融合为一个SuperKernel进行计算。

```python
torch.npu.super_kernel_scope_begin(scope_name: str)
待融合的算子操作
torch.npu.super_kernel_scope_end(scope_name: str)
```

scope_name：表示该范围内算子融合后的SuperKernel名称，相同的scope\_name代表相同的融合范围，由用户控制。若传入None，则该范围内的算子不进行SuperKernel融合。


2. 通过npugraph\_ex的options配置开启SuperKernel融合优化，

```python
import torch
import torch_npu

opt_model = torch.compile(model, backend="npugraph_ex", options={"super_kernel_optimize": True, "super_kernel_optimize_options": dict, "super_kernel_debug_options": dict}, dynamic=False, fullgraph=True)
```

（配置可以借鉴[gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/npugraph_ex.md](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/npugraph_ex.md))
