# Plücker 密集射线 v2

历史版本，已由 `RECAM_MULTIVIEW_V3.md` 更正：删除ReCamMaster projector超出了只替换相机条件分支的范围。当前代码恢复官方ReCamMaster的单位初始化projector；v2代码保存在父目录plucker_dense_v2_before_restore.zip。以下仅描述历史v2。

依据用户2026-11-16指令，对齐PlenopticDreamer公开实现的射线patch、相机投影及初始化，删除额外ReCamMaster projector。仅独立multiview分支；旧训练未改、未开始新正式训练。v1归档在父目录plucker_sparse_v1_before_dense.zip，旧验收记录保留。

## 固定参考

NVIDIA cosmos-transfer2.5 commit `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`：

- `cosmos_transfer2/_src/imaginaire/modules/camera.py`：逐像素中心、单位射线、顺序(moment,direction)。
- `cosmos_transfer2/_src/predict2/camera/utils.py`：16×16射线展平，不平均。
- `cosmos_transfer2/_src/predict2/camera/configs/multiview_camera/conditioner.py`：patch_spatial=16，camera_patch_average=False。
- `cosmos_transfer2/_src/predict2/camera/networks/dit_multiview_camera_ar.py`：每block独立无bias线性相机投影，截断正态初始化，归一化与调制之后加到自注意力输入。

## 当前实现

每个处理后图像像素中心(i+0.5,j+0.5)计算射线，按16×16区域展平为1536维，空间顺序patch行、patch列、通道，射线通道为[o×d,d]。384×672对应24×42 token，和Wan stride8 VAE + 2×2 DiT patch一致。几何计算float32，进入网络后转模型dtype，不使用平均/插值或稀疏采样。密集射线临时内存比v1增加64倍；后续训练需实测峰值，必要时分块计算但不改变几何定义。不需重新生成视频VAE缓存。

每block `Linear(1536,D,bias=False)`；`trunc_normal_(std=1/sqrt(D),a=-3std,b=3std)`，Wan1.3B中D=1536。不是零初始化，也没有额外全局相机token。

前向为：

```python
attn_input = modulate(norm1(x), shift, scale) + camera_encoder(ray_patch)
x = x + gate * self_attention(attn_input, rope)
```

删除v1的 `projectors` ModuleList；保留self-attention本身已有的输出投影。额外projector来源是官方ReCamMaster的train_recammaster.py第292行（Linear(dim,dim)，第295行单位初始化），v1沿用该设计；PlenopticDreamer对应block没有这层，因此v2删除。

其他Wan适配保持：各视角同步RoPE、源/目标role embedding、公共参考相机、空文本交叉注意力、只预测目标；这些并不声称逐项复制Cosmos。时间仍采用原始帧0,4,...80代表21latent时刻。默认参考最小ID源首帧。去除1–4源的硬上限，模型至少1源，受内存限制；数据适配器受每场景10相机且目标/源不可重复的实际数据限制。同批仍需相同源数量。

相机编码输入由24变1536，删除projector，v1适配器检查点不兼容，禁止strict=False静默载入。重新从原始Wan构建新模型。

## 验证

`test_multiview_plucker.py --assets ../assets.json --official-camera ../official_camera_reference.py --output ../plucker_dense_v2_acceptance.json`。

直接加载固定提交的官方Camera.get_plucker_rays作数值对照，验证通道与patch展平一致；另检查截断正态参数、无bias/额外projector、源排序、1/2/4/5源、小模型梯度/优化器RNG恢复、四焦距真实缓存和空文本以及真实缓存子集3步flowmatching。

`prepare_multiview_model.py`检查原始Wan权重载入、密集投影初始化及参数有限性，回执plucker_dense_v2_base_verification.json。均为CPU；不等于全尺寸GPU训练或质量评估。后续仍需课程采样器、DDP及完整训练评估集成。
