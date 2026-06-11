# U-Net 道路场景图像分割与清晰度增强实验报告

## 1. 实验目的

本实验以 U-Net 为主题，面向道路应用场景完成两项内容：

1. 道路场景语义分割：将真实驾驶场景图像划分为背景、道路、车辆和障碍物四类。
2. 图像清晰度增强展示：对当前 `img/` 目录下的三张低清图进行无参考视觉增强，并展示增强前后的效果。

实验重点是比较 U-Net 与其他网络结构在像素级视觉任务中的差异。分割部分比较 U-Net 和 FCN；清晰度增强部分由于没有高清真值图，不进行监督式训练和定量指标评价。

道路场景具有明确应用价值。道路区域、车辆和障碍物识别是自动驾驶、辅助驾驶、道路巡检和移动机器人导航中的基础视觉任务，因此适合作为课程大作业案例。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows |
| Python | 3.13.13 |
| 深度学习框架 | PyTorch |
| 主要依赖 | numpy, Pillow, matplotlib, tqdm |
| 训练设备 | CUDA |
| 分割数据集 | CamVid Tiny |
| 图像尺寸 | 96 x 96 |
| 分割训练集规模 | 80 |
| 分割验证集规模 | 20 |
| Batch Size | 8 |
| 随机种子 | 42 |

## 3. 数据集与任务构造

### 3.1 CamVid 道路场景分割数据

本实验使用公开道路场景数据集 CamVid 的小规模真实子集 `camvid_tiny`。CamVid 图像来自驾驶视角，提供像素级语义标签，适合道路场景理解任务。

为了贴合课程项目需求，本实验将 CamVid 原始类别合并为四类：

| 项目类别 | CamVid 原始类别示例 | 含义 |
|---|---|---|
| background | Sky, Building, Tree, Wall 等 | 天空、建筑、树木、墙面等背景 |
| road | Road, RoadShoulder, Sidewalk, LaneMkgsDriv, LaneMkgsNonDriv | 路面、路肩、人行道和车道线 |
| vehicle | Car, SUVPickupTruck, Truck_Bus, Train, MotorcycleScooter | 车辆类目标 |
| obstacle | Pedestrian, Bicyclist, TrafficCone, TrafficLight, SignSymbol, Column_Pole, Fence 等 | 行人、骑行者、锥桶、标志、杆、围栏等道路目标 |

真实数据比生成数据更复杂，包含光照变化、遮挡、模糊、透视变化和类别不均衡，因此指标会低于简单合成数据，但更有实际应用意义。

### 3.2 清晰度增强输入

当前清晰度增强使用 `img/1.jpg`、`img/2.jpg`、`img/3.jpg`。这些图片本身已经是模糊低清图，没有对应的高清大图作为标准答案。

因此本实验不再采用“先人为模糊，再训练模型恢复”的方式，也不计算 MSE、PSNR、SSIM 等需要高清参考图的指标。该部分只做无参考视觉增强展示，输出增强前后的对比图。

## 4. 模型与方法设计

### 4.1 U-Net 分割模型

U-Net 使用编码器-解码器结构。编码器通过卷积和池化提取高层语义信息；解码器通过上采样恢复空间分辨率。skip connection 将编码器中同尺度的浅层特征拼接到解码器中。

skip connection 是 U-Net 的关键。它可以把浅层边缘、位置和纹理信息传回解码阶段，使模型在像素级预测中更容易恢复边界、小目标和道路轮廓。

### 4.2 FCN 分割基线

FCN 同样使用卷积、池化和上采样完成像素级分类，但没有 U-Net 的同尺度跳跃连接。因此它更依赖瓶颈层之后的低分辨率特征，容易损失细粒度边界信息。

### 4.3 清晰度增强方法

清晰度增强部分采用传统图像处理流程，而不是监督式深度学习模型：

1. 使用 Lanczos 插值按比例放大图像。
2. 转为灰度图并自动拉伸对比度。
3. 使用中值滤波、形态学滤波和小连通域清理减少背景噪点。
4. 提白背景、保留主要灰度线条。
5. 使用 Unsharp Mask 和温和对比度增强突出轮廓。

这种方法适合当前没有高清真值图的情况。它能让线条更明显、对比度更强，但不能保证恢复出原本不存在的真实细节。

## 5. 训练与评价指标

分割任务使用带类别权重的 Cross Entropy Loss。由于 CamVid 中车辆和障碍物像素占比较低，类别权重可以减少模型只偏向背景和道路的风险。

分割评价指标包括：

| 指标 | 含义 |
|---|---|
| mIoU(all) | 所有类别 IoU 的平均值 |
| mIoU(foreground) | road、vehicle、obstacle 三个前景类别 IoU 的平均值 |
| Pixel Accuracy | 像素分类准确率 |
| Class IoU | 单个类别的交并比 |

清晰度增强部分只做视觉展示，不使用全参考指标。原因是当前输入图片没有高清真值图，强行计算 MSE、PSNR、SSIM 会让实验结论不成立。

## 6. 实验结果

### 6.1 分割结果

| 模型 | mIoU(all) | mIoU(foreground) | Pixel Accuracy | 参数量 |
|---|---:|---:|---:|---:|
| U-Net | 0.6439 | 0.5774 | 0.8852 | 117,732 |
| FCN | 0.6038 | 0.5353 | 0.8643 | 35,894 |

各类别 IoU 如下：

| 类别 | U-Net IoU | FCN IoU |
|---|---:|---:|
| background | 0.8433 | 0.8095 |
| road | 0.9154 | 0.8932 |
| vehicle | 0.6039 | 0.5119 |
| obstacle | 0.2129 | 0.2008 |

分割结果图：

![Segmentation examples](outputs_camvid/figures/segmentation_examples.png)

分割验证损失：

![Segmentation loss](outputs_camvid/figures/segmentation_loss.png)

结果显示，U-Net 在整体 mIoU、前景 mIoU 和像素准确率上都优于 FCN。车辆类 IoU 提升较明显，说明 skip connection 对恢复小目标和边界有帮助。障碍物类 IoU 较低，主要原因是该类别由多个小类别合并而来，目标形态差异大、像素占比低。

### 6.2 清晰度增强结果

清晰度增强结果图：

![Restoration examples](outputs_camvid/figures/restoration_examples.png)

增强后图像背景噪点明显减少，线条更黑，轮廓更清楚，对比度更高，适合在汇报中展示“低清输入经过增强后的可视化变化”。但该结果不能解释为严格意义上的高清还原，因为没有高清真值图作为标准。

## 7. 结果分析

分割实验表明，U-Net 的优势主要体现在小目标和边界恢复上。道路场景中，车辆和障碍物面积较小，如果只依赖深层低分辨率特征，上采样时容易丢失位置细节。U-Net 通过 skip connection 将浅层空间信息传回解码器，因此能够更准确地恢复目标边界。

FCN 的参数量更少，但由于缺少同尺度特征融合，它在车辆和障碍物类别上的表现弱于 U-Net。这说明在道路场景分割任务中，结构设计比单纯参数量更重要。

清晰度增强部分说明，在没有高清真值图的条件下，采用传统图像处理做展示更合理。它可以改善视觉观感，但不能提供严格的定量结论。若后续需要做完整的图像复原实验，应补充成对的低清/高清数据集。

## 8. 局限性与改进方向

当前实验使用 CamVid Tiny，优点是下载快、复现稳定；不足是数据规模较小，障碍物类别样本不足。完整 CamVid、Cityscapes 或 BDD100K 能提供更充分的数据，但下载和训练成本更高。

当前图像分辨率为 96 x 96，模型规模也较小。提高分辨率、增加训练轮数和模型宽度，有可能进一步提升结果。

进一步改进方向包括：

- 使用完整 CamVid 训练更多样本。
- 使用 Cityscapes 或 BDD100K 做更大规模道路场景分割。
- 将 `obstacle` 拆成 pedestrian、sign、pole、traffic cone 等更细类别。
- 尝试 DeepLabV3、SegNet、U-Net++ 等分割网络。
- 为图像复原部分补充成对低清/高清图片，再进行严格定量评价。

## 9. 复现实验方法

在仓库根目录运行：

```bash
python run_road_experiments.py --seg-dataset camvid_tiny --epochs 25 --train-count 80 --val-count 20 --batch-size 8 --size 96 --output-dir outputs_camvid
```

运行完成后会生成：

| 文件 | 说明 |
|---|---|
| `outputs_camvid/metrics.json` | 实验配置、分割指标和训练历史 |
| `outputs_camvid/figures/segmentation_examples.png` | 分割可视化对比 |
| `outputs_camvid/figures/segmentation_loss.png` | 分割验证损失曲线 |
| `outputs_camvid/figures/restoration_examples.png` | 清晰度增强可视化对比 |

## 10. 总结

本实验使用公开 CamVid 道路场景数据完成了 U-Net 图像分割实验，并用当前 `img/` 目录下的三张低清图完成了清晰度增强展示。分割结果显示，U-Net 在 mIoU、前景 mIoU 和像素准确率上均优于 FCN，尤其在车辆和障碍物这类小目标上更稳定。

整体来看，U-Net 的编码器-解码器结构和 skip connection 适合需要空间细节恢复的视觉任务。清晰度增强部分由于没有高清真值图，只作为展示性补充；严格图像复原实验需要额外准备成对数据。
