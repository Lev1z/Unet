# U-Net 道路场景图像分割与清晰度还原实验报告

## 1. 实验目的

本实验以 U-Net 为主题，面向道路应用场景完成两类任务：

1. 道路场景语义分割：将图像像素划分为背景、道路、车辆和障碍物四类。
2. 图像清晰度还原：将低清、模糊、含噪图像恢复为更接近清晰目标图的结果。

实验重点不是单独训练一个模型，而是比较 U-Net 与其他网络结构的效果差异。分割部分比较 U-Net 和 FCN；还原部分比较残差式 Restoration U-Net 和 Plain CNN。

道路场景具有明确应用价值。道路区域、车辆和障碍物识别是自动驾驶、辅助驾驶、道路巡检和移动机器人导航中的基础视觉任务，因此比单纯几何图形或无应用背景的数据更适合作为课程大作业案例。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| Python | 3.13.13 |
| 深度学习框架 | PyTorch |
| 主要依赖 | numpy, Pillow, matplotlib, scikit-image, tqdm |
| 训练设备 | CUDA |
| 图像尺寸 | 96 x 96 |
| 分割训练集规模 | 768 |
| 分割验证集规模 | 192 |
| 还原训练样本数 | 96 |
| 还原验证样本数 | 12 |
| Batch Size | 32 |
| 随机种子 | 42 |

## 3. 数据集与任务构造

### 3.1 道路场景分割数据

本实验使用本地生成的道路场景数据。每张图像包含道路、车辆、路障、锥桶、箱体、背景建筑和天空等元素，同时生成对应的像素级语义标签。

类别定义如下：

| 类别 ID | 类别 | 说明 |
|---:|---|---|
| 0 | background | 天空、建筑、远景背景 |
| 1 | road | 可通行道路或地面 |
| 2 | vehicle | 道路车辆 |
| 3 | obstacle | 锥桶、路障、箱体等障碍物 |

这类数据虽然不是大型真实公开数据集，但它有两个优点：第一，可以稳定生成像素级标签；第二，任务目标与道路感知应用一致，适合验证 U-Net 对小目标和边界的恢复能力。

### 3.2 图像清晰度还原数据

图像还原实验优先读取 `img/` 目录下最多 3 张图片作为清晰目标图。脚本会对清晰图执行退化操作，生成模型输入：

- 降采样和上采样，模拟低分辨率；
- Gaussian Blur，模拟镜头模糊或运动模糊；
- 随机噪声，模拟采集噪声。

当前工作区暂未提供真实 `img/` 图片，因此本次结果使用 3 张生成道路图作为占位源图。后续补入真实图片后，可以不改代码直接重跑实验。

## 4. 模型设计

### 4.1 U-Net 分割模型

U-Net 采用编码器-解码器结构。编码器通过卷积和池化逐步提取高层语义信息；解码器通过上采样恢复空间分辨率；skip connection 将编码器中同尺度的浅层特征拼接到解码器中。

skip connection 是 U-Net 的关键。它可以把浅层边缘、位置和纹理信息补回解码阶段，使模型在像素级预测任务中更容易恢复边界和小目标。

### 4.2 FCN 分割基线

FCN 同样使用卷积、池化和上采样完成像素级分类，但没有 U-Net 的同尺度跳跃连接。因此它更依赖瓶颈层之后的上采样特征，容易损失细粒度边界信息。

### 4.3 Restoration U-Net

图像还原模型采用残差式 U-Net。模型不是直接生成整张清晰图，而是学习退化图到清晰图之间的残差修正量：

```text
restored image = degraded image + predicted residual
```

这种设计适合图像复原任务。低清输入中仍然包含整体结构，模型只需要重点学习去噪、锐化和局部修正。

### 4.4 Plain CNN 基线

Plain CNN 由连续卷积层组成，不做显式下采样和上采样。它参数量更少，适合作为局部像素映射基线。与 Restoration U-Net 对比，可以观察多尺度结构和残差学习是否带来收益。

## 5. 训练与评价指标

分割任务使用 Cross Entropy Loss。评价指标包括：

| 指标 | 含义 |
|---|---|
| mIoU(all) | 所有类别 IoU 的平均值 |
| mIoU(foreground) | road、vehicle、obstacle 三个前景类别 IoU 的平均值 |
| Pixel Accuracy | 像素分类准确率 |
| Class IoU | 单个类别的交并比 |

图像还原任务使用 MSE Loss。评价指标包括：

| 指标 | 含义 |
|---|---|
| MSE | 像素均方误差，越低越好 |
| PSNR | 峰值信噪比，越高越好 |
| SSIM | 结构相似度，越高越好 |

## 6. 实验结果

### 6.1 分割结果

| 模型 | mIoU(all) | mIoU(foreground) | Pixel Accuracy | 参数量 |
|---|---:|---:|---:|---:|
| U-Net | 0.9864 | 0.9822 | 0.9984 | 117,732 |
| FCN | 0.9315 | 0.9103 | 0.9916 | 35,894 |

各类别 IoU 如下：

| 类别 | U-Net IoU | FCN IoU |
|---|---:|---:|
| background | 0.9990 | 0.9949 |
| road | 0.9967 | 0.9787 |
| vehicle | 0.9659 | 0.8815 |
| obstacle | 0.9839 | 0.8708 |

分割结果图：

![Segmentation examples](outputs_road/figures/segmentation_examples.png)

可以看到，U-Net 对道路边界、车辆和障碍物的位置恢复更稳定。FCN 对大区域背景和道路也能完成基本分割，但在车辆和障碍物这类小目标上 IoU 明显低于 U-Net。

### 6.2 清晰度还原结果

| 模型/输入 | MSE | PSNR | SSIM | 参数量 |
|---|---:|---:|---:|---:|
| Degraded input | 0.002626 | 25.8711 | 0.6207 | - |
| Restoration U-Net | 0.002027 | 26.9868 | 0.6453 | 117,715 |
| Plain CNN | 0.003586 | 24.4707 | 0.5937 | 20,259 |

还原结果图：

![Restoration examples](outputs_road/figures/restoration_examples.png)

Restoration U-Net 相比退化输入降低了 MSE，并将 PSNR 从 25.8711 dB 提升到 26.9868 dB，SSIM 也从 0.6207 提升到 0.6453。Plain CNN 的输出更平滑，但细节和结构保留不足，三个指标均低于 Restoration U-Net。

## 7. 结果分析

分割实验表明，U-Net 的优势主要体现在小目标和边界恢复上。道路场景中，车辆和障碍物面积较小，如果只依赖深层低分辨率特征，上采样时容易丢失位置细节。U-Net 通过 skip connection 将浅层空间信息传回解码器，因此能够更准确地恢复目标边界。

还原实验表明，残差式 U-Net 比 Plain CNN 更适合该任务。图像还原不是从零生成图像，而是在已有退化图基础上修正模糊和噪声。残差学习保留了输入图的整体结构，多尺度 U-Net 则提供了更大的感受野和更强的上下文建模能力。

## 8. 局限性与改进方向

当前分割数据为本地生成道路场景，优点是标签稳定、复现实验方便；不足是真实道路图像中的光照、遮挡、透视变化和复杂目标更多，难度更高。

当前清晰度还原实验暂未使用真实 `img/` 图片，而是使用 3 张生成道路图占位。后续应补充 2-3 张真实道路或场景图片，重新生成还原结果，使报告更贴近实际应用。

进一步改进方向包括：

- 使用 CamVid、Cityscapes 或 BDD100K 等真实道路数据集；
- 增加训练样本量、图像分辨率和训练轮数；
- 尝试更强的分割网络，例如 DeepLabV3、SegNet、U-Net++；
- 在还原任务中加入 L1 Loss、SSIM Loss 或感知损失；
- 增加更多真实低清图片，比较不同退化类型下的恢复效果。

## 9. 复现实验方法

在仓库根目录运行：

```bash
python run_road_experiments.py --epochs 25 --train-count 768 --val-count 192 --restore-train-count 96 --restore-val-count 12 --batch-size 32 --size 96 --output-dir outputs_road
```

运行完成后会生成：

| 文件 | 说明 |
|---|---|
| `outputs_road/metrics.json` | 实验配置、指标和训练历史 |
| `outputs_road/figures/segmentation_examples.png` | 分割可视化对比 |
| `outputs_road/figures/segmentation_loss.png` | 分割验证损失曲线 |
| `outputs_road/figures/restoration_examples.png` | 图像还原可视化对比 |
| `outputs_road/figures/restoration_loss.png` | 还原验证损失曲线 |

## 10. 总结

本实验围绕道路场景完成了 U-Net 图像分割和图像清晰度还原两项任务。分割结果显示，U-Net 在 mIoU、前景 mIoU 和像素准确率上均优于 FCN，尤其在车辆和障碍物小目标上优势明显。还原结果显示，Restoration U-Net 相比退化输入和 Plain CNN 均取得更好的 MSE、PSNR 和 SSIM。

整体来看，U-Net 的编码器-解码器结构和 skip connection 适合需要空间细节恢复的视觉任务；残差式 U-Net 也可以迁移到图像复原任务中，用于改善低清、模糊和含噪图像的质量。
