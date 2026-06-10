# U-Net 图像分割与图像清晰度还原实验报告

## 1. 实验目的

本实验以 U-Net 为主题，完成两个相关任务：

1. 图像分割：比较 U-Net 与无跳跃连接的轻量 FCN 在目标区域分割上的效果差异。
2. 图像清晰度还原：比较 Residual U-Net 与普通卷积神经网络 Plain CNN 在模糊、降采样、噪声退化图像上的还原效果差异。

实验重点不是追求大型公开数据集上的最高分，而是通过一个可复现实验观察 U-Net 的核心结构特征：编码器-解码器结构、跳跃连接、多尺度语义与浅层细节融合。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| Python | 3.13.13 |
| 深度学习框架 | PyTorch |
| 主要依赖 | numpy, Pillow, matplotlib, scikit-image, tqdm |
| 训练设备 | CUDA |
| 图像尺寸 | 64 x 64 |
| 训练集规模 | 1024 张 |
| 验证集规模 | 128 张 |
| 训练轮数 | 50 |
| Batch Size | 64 |
| 随机种子 | 42 |

## 3. 数据集构造

本实验使用脚本生成合成图像数据，原因是课程作业需要能在本地稳定复现，避免外部数据下载失败或路径配置问题。每张图像由渐变噪声背景和若干随机几何目标组成，目标形状包括圆形、矩形和三角形。

分割任务中，输入为 RGB 合成图像，标签为几何目标的二值掩膜。

图像还原任务中，先生成清晰图像，再对其进行降采样、上采样、Gaussian Blur 和随机噪声扰动，得到退化图像。模型输入退化图像，学习恢复到原始清晰图像。

## 4. 模型设计

### 4.1 U-Net

U-Net 使用两层下采样编码器、一个瓶颈层和两层上采样解码器。解码阶段通过跳跃连接拼接同尺度编码器特征，使模型同时获得高级语义信息和浅层空间细节。

在分割任务中，U-Net 输出 1 通道 mask logits；在图像还原任务中，本实验使用 Residual U-Net 输出退化图像到清晰图像之间的残差修正量。

参数量：

| 任务 | 模型 | 参数量 |
|---|---:|---:|
| 分割 | U-Net | 66,469 |
| 还原 | Residual U-Net | 117,715 |

### 4.2 对比模型

分割对比模型为轻量 FCN。它同样使用卷积、池化和上采样，但没有 U-Net 的同尺度跳跃连接，因此在恢复边界和小目标时更依赖瓶颈后的上采样特征。

图像还原主模型为 Residual U-Net。它保留 U-Net 的编码器-解码器和跳跃连接结构，同时不直接从零生成清晰图像，而是学习退化图像到清晰图像之间的残差修正：

```text
restored image = degraded image + predicted residual
```

图像还原对比模型为 Plain CNN，由连续卷积层组成，不进行显式下采样。它参数较少，保留完整空间分辨率，适合作为局部像素映射基线。

参数量：

| 任务 | 模型 | 参数量 |
|---|---:|---:|
| 分割 | FCN | 23,049 |
| 还原 | Plain CNN | 20,259 |

## 5. 训练与评价指标

分割任务使用 BCEWithLogitsLoss 与 Dice Loss 的组合损失：

```text
Loss = 0.5 * BCE + (1 - Dice)
```

评价指标包括：

| 指标 | 含义 |
|---|---|
| mIoU | 预测区域与真实区域的平均交并比 |
| Dice | 分割任务常用重叠指标，对小目标较敏感 |
| Pixel Accuracy | 像素级分类准确率 |

图像还原任务使用 MSE Loss。评价指标包括：

| 指标 | 含义 |
|---|---|
| MSE | 像素均方误差，越低越好 |
| PSNR | 峰值信噪比，越高越好 |
| SSIM | 结构相似性，越高越好 |

## 6. 实验结果

### 6.1 图像分割结果

| 模型 | mIoU | Dice | Pixel Accuracy |
|---|---:|---:|---:|
| U-Net | 0.9863 | 0.9930 | 0.9981 |
| FCN | 0.9695 | 0.9844 | 0.9958 |

分割示例：

![Segmentation examples](outputs/figures/segmentation_examples.png)

从结果看，U-Net 在 mIoU 和 Dice 上均优于轻量 FCN。可视化结果中，U-Net 对小目标和多个目标同时出现的场景更稳定，边界形状整体更接近标签。原因在于跳跃连接将编码器早期的空间细节直接传递给解码器，减少了连续池化和上采样导致的位置信息损失。

训练损失曲线：

![Segmentation loss](outputs/figures/segmentation_loss.png)

### 6.2 图像清晰度还原结果

| 模型/输入 | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| Degraded input | 0.004030 | 24.1474 | 0.4195 |
| Residual U-Net | 0.001163 | 29.4981 | 0.6751 |
| Plain CNN | 0.001676 | 27.9306 | 0.6598 |

还原示例：

![Restoration examples](outputs/figures/restoration_examples.png)

图像还原结果显示，Residual U-Net 相比退化输入显著降低 MSE，并将 PSNR 从 24.1474 dB 提升到 29.4981 dB，SSIM 也从 0.4195 提升到 0.6751。与 Plain CNN 相比，Residual U-Net 在 MSE、PSNR 和 SSIM 上均取得更好结果，说明残差学习能帮助 U-Net 更稳定地完成图像复原。

训练损失曲线：

![Restoration loss](outputs/figures/restoration_loss.png)

## 7. 结果分析

U-Net 在分割任务上表现更好，核心原因是语义分割需要同时判断“目标是什么”和“目标在哪里”。编码器通过下采样获得更大感受野，解码器恢复空间分辨率，而跳跃连接补充浅层边缘和位置信息。因此，U-Net 对边界、小目标和多目标区域更友好。

图像还原任务中，Residual U-Net 的表现优于 Plain CNN。原因在于残差学习避免了从零重建整张图像的难度，模型只需要学习退化输入与清晰图像之间的修正量；同时 U-Net 的跳跃连接仍然可以保留浅层空间细节，多尺度编码器-解码器结构可以帮助模型恢复边缘和结构。

因此，本实验可以得到两个结论：

1. U-Net 的跳跃连接对图像分割非常有效，尤其适合需要精确定位的像素级预测任务。
2. 对图像清晰度还原任务，残差式 U-Net 能同时利用输入图像的原始结构和多尺度特征，在本实验中优于普通 CNN。

## 8. 局限性与改进方向

本实验使用的是合成几何图像，便于快速复现和观察结构差异，但与真实医学图像、遥感图像或自然图像仍有差距。后续可以替换为真实数据集，例如 Oxford-IIIT Pet、Carvana、ISBI 细胞分割数据集等。

图像还原部分当前使用 MSE Loss，容易得到平滑结果。后续可以加入 L1 Loss、SSIM Loss、感知损失，或者使用残差学习结构，例如 DnCNN、RED-Net、ResUNet 等。

当前模型规模较小，训练轮数也较少。若进一步增加数据量、训练 epoch 和模型宽度，结果可能发生变化。

## 9. 复现实验方法

在当前目录下执行：

```bash
python run_experiments.py --epochs 50 --train-count 1024 --val-count 128 --batch-size 64 --size 64
```

运行完成后会生成：

| 文件 | 说明 |
|---|---|
| `outputs/metrics.json` | 实验配置、指标和训练历史 |
| `outputs/figures/segmentation_examples.png` | 分割可视化对比图 |
| `outputs/figures/restoration_examples.png` | 图像还原可视化对比图 |
| `outputs/figures/segmentation_loss.png` | 分割验证损失曲线 |
| `outputs/figures/restoration_loss.png` | 还原验证损失曲线 |

## 10. 总结

本实验围绕 U-Net 完成了图像分割和图像清晰度还原两个任务。分割实验中，U-Net 相比轻量 FCN 获得更高 mIoU 和 Dice，验证了跳跃连接在像素级定位任务中的优势。图像还原实验中，Residual U-Net 相比退化输入和 Plain CNN 都取得更好的整体指标，说明在图像复原任务中加入残差学习后，U-Net 也能有效恢复图像结构和边缘细节。
