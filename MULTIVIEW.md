# ReCamMaster 多视角分支

本分支基于官方 Wan2.1-T2V-1.3B ReCamMaster，支持可变数量的同步源视频。当前课程覆盖1～4源；网络没有将源数硬编码为4，但更多源的显存、吞吐和效果未经本实验验证。

本文件是多视角分支的当前总入口。网络设计见[架构说明](docs/architecture.md)；[历史方案](docs/history/)仅用于追溯，不作为当前配置。

## 模型

- 多个源latent与目标latent联合自注意力，各视频使用相同的帧时间坐标。
- 密集Plücker射线按16×16像素patch展平为1536维，经每block的线性相机编码器映射后注入对应patch。
- 以第一个源的第一帧为坐标参考，不进行场景尺度归一化。
- 保留ReCamMaster额外projector及官方Wan实现的单位初始化。
- 空文本条件保留文本交叉注意力。自注意力选择性checkpoint缓存减少反向重算。

模型及几何实现见 `diffsynth/models/wan_video_multiview.py`、`plucker_geometry.py`；架构及初始化依据见[当前架构说明](docs/architecture.md)。

## 当前训练源码快照（2026-11-20）

训练入口及辅助模块来自正在运行的 `mv20k_bs16_lr4e5_20261119` 冻结源码，而非早期准备版。8个rank、单卡microbatch1、梯度累积2，有效batch16；从共同父5000步完整恢复模型、Adam与RNG。学习率4e-5保持至8000步，按全局步数余弦衰减至20000步的1e-5；含5200步一次性稳定性检查。250步保存、500步固定展示。

`train_multiview.py`是本实验的恢复训练入口，不是无需配置的一键复现脚本。运行前需要实验protocol、合法的checkpoint identity、Wan基座、空文本特征、几何预处理结果、VAE缓存及课程统计。工作区根需要 `assets.json`、`planning_20k_v1/train_geometry_metrics.npz`；运行目录需要 `protocol.json`。脚本通过 `--root`、`--run`、`--resume`显式指定这些位置，不包含用户机器上的配置或权重。

`multiview_data.py`负责缓存适配，`curriculum_sampling.py`负责源数量和视角差异课程，`evaluate_multiview.py`负责固定生成探针，`benchmark_multiview.py`用于独立测量。训练数据与完整预处理资产应按项目实验协议另行准备。

本提交只发布网络、数据适配、课程、训练与固定探针源码及测试；工作区之外的任务中心、飞书配置、机器监控、结果看板、扩展指标调度不包含在此分支中。没有上传任何训练数据、模型权重、生成视频或私密凭证。已在真实多卡实验验证的历史结果不代表论文严格复现或未见环境泛化成功。

## CPU结构检查

安装原仓库依赖后运行：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python test_multiview_plucker.py --output /tmp/multiview-check.json
```

该检查使用小模型，验证相机编码、同步时间RoPE、多源排列和梯度等结构性质；不启动正式训练，也不替代全尺寸质量评估。`--assets`可附加已有真实缓存的适配检查。
