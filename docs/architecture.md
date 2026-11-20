# 当前架构：ReCamMaster 多视角输入

[返回多视角总入口](../MULTIVIEW.md)。本页说明当前网络结构；训练配置与使用要求以总入口为准。历史 V1/V2 已归档到 [history](history/)。

用户目标是将ReCamMaster扩展为多视角输入，不是将其替换成PlenopticDreamer。以官方ReCamMaster Wan版为结构基线，只在明确指定的相机条件分支借鉴PlenopticDreamer。2026-11-16纠正v2范围错误，恢复projector。

## 当前结构

源视频由单个扩展为可变数量，每个视频独立VAE编码后沿时间/token轴拼接；各视角RoPE共享同步时间，源/目标相机条件各自绑定对应patch。维持空文本及交叉注意力。加入的源/目标role embedding保持当前版本，源排列不改变角色编码。

相机分支：密集16×16像素射线patch，不平均，1536维(moment,direction)；每block独立Linear(1536,D,bias=False)，截断正态std=1/sqrt(D)、±3std。此部分按PlenopticDreamer公开实现对齐；裁剪内参和代表时刻说明见[历史 V2](history/PLUCKER_V2.md)。2026-11-16按用户要求，默认参考改为输入列表第一个源的第一帧（不再取最小相机编号），新增源不得改变锚点。平移米制，不做场景尺度归一化；显式reference_camera接口保留。

ReCamMaster残差保持：

```python
attn_input = modulate(norm1(x), shift, scale) + camera_encoder(ray_patch)
x = x + gate * projector(self_attention(attn_input, rope))
```

projector是额外Linear(D,D)，不是相机投影，也不是self-attention内置输出投影。初始化weight=I、bias=0，与官方Wan版train_recammaster.py:292–296一致，参与训练。原始Wan权重加载后才构建这些适配模块，不载入旧单视角适配器。

## projector的证据与解释

官方代码明确在自注意力输出后加入该层并训练它。其可观察作用是对融合后的特征作可学习通道线性变换，控制注入残差的内容；单位初始化使新增层初始不改变该通路输出。可以理解为任务适配的再投影，但不能将这一解释说成作者已证明的独立收益。

论文§5.1提到projector零初始化，Wan开源代码实际单位初始化，二者不一致。本项目已选择Wan官方代码路线，因此采用单位初始化。所检查正文未给出单独移除projector的消融或详细动机；不据此断言它无用。它与attention内部输出线性层之间无非线性，推理时可合并为仿射映射，但训练参数化/优化路径不同，不能据此视为可随意删除。

## 变更范围约束

保留ReCamMaster模块为默认。PlenopticDreamer仅作为指定相机分支及已讨论采样方法的参考；其余结构变化必须说明必要性并作为独立实验，不因另一篇论文没有某模块而自动删除。原query复现不变；用户后续授权启动vllm1多视角正式训练，详见父目录TRAINING_PLAN.md及实验mv20k_20261116。

保留v1和v2源码归档、历史验收，v3单独回执plucker_dense_v3_acceptance.json与plucker_dense_v3_base_verification.json。测试确认projector单位初始化、非零梯度、优化器续跑，及此前密集射线与官方数值一致等检查。CPU小模型不代表全尺寸训练已验证。
