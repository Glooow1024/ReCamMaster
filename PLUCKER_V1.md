# Plücker 多视角条件 v1

历史版本：当前实现已按用户要求升级为密集射线v2，见 `PLUCKER_V2.md`。v1源码保存在父目录 `plucker_sparse_v1_before_dense.zip`；以下内容仅描述历史方案。

独立 multiview 分支；官方脚本/原复现工作区不变。没有开始全尺寸训练。

## 实现

- `diffsynth/models/plucker_geometry.py`：裁剪内参、共同参考坐标、射线及空间patch分组。
- `diffsynth/models/wan_video_multiview.py`：包装原始Wan DiT，1–4个干净源视频、1个带噪目标，源和目标沿时间序列拼接做联合注意力，仅返回目标预测。
- `multiview_data.py`：只读复用几何NPZ、现有VAE latent和一份空文本特征。显式给出目标/源相机ID，不静默换样本。
- `prepare_multiview_model.py`：仅从原始Wan权重初始化，不加载已训练的单视角ReCamMaster适配器。
- `test_multiview_plucker.py`：CPU几何、排序、梯度、检查点及实际缓存对接验证。

## 坐标与特征合同

视频81×384×672 → latent21×48×84 → DiT网格21×24×42。各视角独立编码，禁止跨视角送入同一个VAE时间序列。

数据原始1280×1280，sensor23.76×23.76mm、焦距来自组名。已核对实际缓存代码：等比例缩至672×672，中心裁剪上下各144像素；处理后fx=fy=focal_mm/23.76*672，cx=336，cy=192。使用像素边界坐标，像素中心i+0.5。内参基于发布的名义标定，不声称通过渲染深度做过精确重投影标定。

继承官方外参解析、轴变换和cm→m。原始转换矩阵可能为反射坐标基，先对所有视角共同乘参考相机的逆，得到正交且det=+1的相对旋转，检查后再计算叉积。参考默认最小ID的源相机第一帧，可显式指定；源输入交换顺序不会改变默认参考。不同源集合比较时应显式固定同一个共有源作为参考。

每个latent单元中心根据处理后K计算单位方向d，相机中心o与d处于同一参考系，射线顺序为(d,o×d)。每个2×2空间patch保留4条射线，即24维，逐block Linear(24,dim)投影后加到对应patch的self-attention输入。不把相机条件放入全局独立token池。

时间采用原始帧0,4,...80的代表位姿；这是对时间压缩的近似，不是对VAE完整时间感受野的精确编码。RoPE每个视角使用相同的latent时间0..20；不是视频帧索引0,4,...80，也不是diffusion timestep。

各视角共同计算注意力，目标/源role embedding区分角色；所有源使用同一种role，不使用固定序号embedding。支持同批相同视角数，单双视角训练可按批次混合；目前不支持在同一batch补零模拟缺失视角。

## 初始化与训练边界

相机Linear和role embedding零初始化，新增attention输出projector单位矩阵初始化（bias=0），保持预训练注意力通路；这个选择明确不同于零输出projector。几何本身用float32计算，投影输入转换到网络dtype。单位投影确保初始相机编码器仍可获得梯度，已验证。

提供的可训练策略是self-attention + projector + camera encoder + role embedding；其余Wan参数冻结，文本交叉注意力结构与空文本输入保留。此策略尚未作为正式训练协议启动。

现有接口准备好，但训练课程采样器、完整DDP训练/评估/恢复集成仍待下一阶段；几何NPZ不等于已完成课程采样策略。不得把CPU随机小模型的验证称为1.3B训练跑通或生成质量达标。

## 使用

```python
from prepare_multiview_model import build_model
from multiview_data import load_example
model = build_model('/data/nvs/recammaster/Wan2.1-T2V-1.3B')
batch = load_example('f24_aperture5/scene1', 1, [2,10],
    '/home/users/ly/nvs/recamaster/geometry_v1',
    '/home/3dvision/nvs/recammaster/cache/v2v_empty_384x672_v1')
# Move model and tensors to the authorized device/dtype in the trainer.
# noisy_target and flow target must use the agreed training scheduler.
prediction = model(noisy_target, batch['sources'], timestep,
    batch['context'], batch['ray_tokens'], use_gradient_checkpointing=True)
```

无需缓存完整射线图，按采样到的相机在线计算即可；现有视频缓存可复用。
