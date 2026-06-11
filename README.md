# U-Net 道路场景分割与图像清晰度增强实验

本仓库是机器学习课程大作业项目，主题是 **U-Net 在道路场景图像分割中的应用**，并附带一个无参考的图像清晰度增强展示。项目默认使用公开道路场景数据集 **CamVid Tiny**，任务背景贴近自动驾驶、辅助驾驶、道路巡检和移动机器人导航。

主脚本：`run_road_experiments.py`  
主要输出：`outputs_camvid/`

## 项目目标

本项目包含两部分：

1. **道路场景语义分割**：将真实驾驶场景图像划分为 `background / road / vehicle / obstacle` 四类，并比较 U-Net 与 FCN 的效果差异。
2. **图像清晰度增强展示**：直接使用 `img/1.jpg`、`img/2.jpg`、`img/3.jpg` 三张低清图，展示增强前后的视觉变化。

需要特别说明：当前 `img/` 下的图片本身就是模糊低清图，没有对应的高清真值图。因此清晰度增强部分不做人为模糊、不训练监督式还原模型，也不报告 MSE、PSNR、SSIM 这类需要参考真值的指标，只展示增强结果供主观观察。

## 数据集说明

### CamVid Tiny

分割实验默认使用 `camvid_tiny`，它是 CamVid 的小规模公开子集，下载快、复现稳定，适合课程汇报。脚本首次运行时会自动下载到 `data/`，该目录已被 `.gitignore` 忽略，不会提交到仓库。

脚本也支持完整 CamVid：

```bash
python run_road_experiments.py --seg-dataset camvid --output-dir outputs_camvid_full
```

完整 CamVid 约 599 MB，训练时间更长，但数据更充分。

### 类别映射

| 项目类别 | CamVid 原始类别示例 | 含义 |
|---|---|---|
| `background` | Sky, Building, Tree, Wall, Tunnel 等 | 天空、建筑、树木、墙面等非目标区域 |
| `road` | Road, RoadShoulder, Sidewalk, LaneMkgsDriv, LaneMkgsNonDriv | 路面、路肩、人行道、车道线 |
| `vehicle` | Car, SUVPickupTruck, Truck_Bus, Train, MotorcycleScooter | 车辆类目标 |
| `obstacle` | Pedestrian, Bicyclist, TrafficCone, TrafficLight, SignSymbol, Column_Pole, Fence 等 | 行人、骑行者、锥桶、交通灯、标志牌、杆、围栏等道路目标 |

### 清晰度增强输入

清晰度增强展示默认读取 `img/` 目录下最多三张图片。当前仓库使用：

- `img/1.jpg`
- `img/2.jpg`
- `img/3.jpg`

这三张图没有高清真值，只适合做视觉增强展示。脚本会保留原图比例，进行放大、自动对比度、背景净化、小噪点清理和边缘锐化，并输出前后对比图。

## 项目结构

```text
.
├── run_road_experiments.py
├── report.md
├── README.md
├── requirements.txt
├── img/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── README.md
└── outputs_camvid/
    ├── metrics.json
    └── figures/
        ├── segmentation_examples.png
        ├── segmentation_loss.png
        └── restoration_examples.png
```

| 文件 | 作用 |
|---|---|
| `run_road_experiments.py` | 下载/读取 CamVid，训练 U-Net 和 FCN，生成分割指标与增强展示图 |
| `report.md` | 课程报告正文，可直接作为汇报材料基础 |
| `outputs_camvid/metrics.json` | 当前实验配置、分割指标和训练历史 |
| `outputs_camvid/figures/segmentation_examples.png` | 分割结果可视化对比 |
| `outputs_camvid/figures/segmentation_loss.png` | 分割验证损失曲线 |
| `outputs_camvid/figures/restoration_examples.png` | 清晰度增强前后对比图 |

## 环境准备

```bash
pip install -r requirements.txt
```

主要依赖：

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `matplotlib`
- `tqdm`

脚本会自动检测 CUDA，有可用 GPU 时使用 CUDA，否则使用 CPU。

## 复现实验

在仓库根目录运行：

```bash
python run_road_experiments.py --seg-dataset camvid_tiny --epochs 25 --train-count 80 --val-count 20 --batch-size 8 --size 96 --output-dir outputs_camvid
```

运行完成后会生成：

```text
outputs_camvid/metrics.json
outputs_camvid/figures/segmentation_examples.png
outputs_camvid/figures/segmentation_loss.png
outputs_camvid/figures/restoration_examples.png
```

## 当前实验结果

### 分割总指标

| 模型 | mIoU(all) | mIoU(foreground) | Pixel Accuracy | 参数量 |
|---|---:|---:|---:|---:|
| U-Net | 0.6439 | 0.5774 | 0.8852 | 117,732 |
| FCN | 0.6038 | 0.5353 | 0.8643 | 35,894 |

U-Net 在整体 mIoU、前景 mIoU 和像素准确率上均优于 FCN。对汇报来说，重点可以放在 U-Net 的 skip connection 能把浅层空间细节传回解码器，因此更适合像素级分割和小目标边界恢复。

### 分割类别 IoU

| 类别 | U-Net IoU | FCN IoU |
|---|---:|---:|
| background | 0.8433 | 0.8095 |
| road | 0.9154 | 0.8932 |
| vehicle | 0.6039 | 0.5119 |
| obstacle | 0.2129 | 0.2008 |

车辆类提升比较明显；障碍物类 IoU 偏低，主要原因是该类别由行人、交通标志、杆、围栏等多个小类合并而来，目标形态差异大、像素占比低。

### 分割可视化

![Segmentation examples](outputs_camvid/figures/segmentation_examples.png)

### 分割验证损失

![Segmentation loss](outputs_camvid/figures/segmentation_loss.png)

### 清晰度增强展示

清晰度增强部分直接使用 `img/1.jpg`、`img/2.jpg`、`img/3.jpg`。由于没有高清真值图，本项目不计算 MSE、PSNR、SSIM，也不声称“还原到真实高清图”。输出图只用于展示视觉增强效果。

![Restoration examples](outputs_camvid/figures/restoration_examples.png)

可以观察到，增强后图像线条更黑、边缘更明显，对比度更高，适合在汇报中作为“低清输入的可视化增强”补充展示。但它不是严格的监督式超分辨率或图像复原实验。

## 方法说明

### U-Net

U-Net 使用编码器-解码器结构。编码器通过卷积和池化提取高层语义信息，解码器通过上采样恢复空间分辨率。skip connection 将编码器中的浅层特征拼接到解码器中，使模型在恢复边界、道路轮廓、车辆和障碍物位置时保留更多细节。

### FCN 基线

FCN 同样使用卷积和上采样完成像素级分类，但没有 U-Net 的同尺度跳跃连接。它可以作为结构更简单的基线模型，用来突出 U-Net 在空间细节恢复上的优势。

### 清晰度增强

增强部分采用传统图像处理流程：

1. 使用 Lanczos 插值按比例放大图像。
2. 转为灰度图并自动拉伸对比度。
3. 使用中值滤波、形态学滤波和小连通域清理减少背景噪点。
4. 提白背景、保留主要灰度线条。
5. 使用 Unsharp Mask 和温和对比度增强突出轮廓。

这种方法不需要训练数据和高清标签，适合当前三张低清图片的展示需求。

## 汇报建议

建议三个人按下面方式分工：

| 成员 | 负责内容 |
|---|---|
| 成员 A | U-Net 原理、编码器-解码器结构、skip connection |
| 成员 B | CamVid 数据集、类别映射、训练流程和评价指标 |
| 成员 C | 实验结果、可视化图片、结论与不足 |

PPT 可以按这个顺序组织：

1. 研究背景：道路场景理解在自动驾驶、辅助驾驶和道路巡检中的意义。
2. 数据集：CamVid Tiny，道路场景图像和像素级标签。
3. 任务定义：四类分割目标，以及低清图片的视觉增强展示。
4. 模型结构：U-Net 与 FCN 的区别，重点讲 skip connection。
5. 训练设置：96 x 96 输入、80 张训练图、20 张验证图、类别权重 Cross Entropy。
6. 分割结果：指标表 + `segmentation_examples.png`。
7. 清晰度增强：说明没有高清真值，因此只展示 `restoration_examples.png`。
8. 总结与不足：U-Net 效果更好；CamVid Tiny 数据较小；障碍物类别仍较难。

## 后续改进

- 使用完整 CamVid 训练更多样本。
- 换用 Cityscapes 或 BDD100K 做更大规模道路场景分割。
- 将 `obstacle` 拆成 pedestrian、traffic sign、pole、traffic cone 等更细类别。
- 尝试 DeepLabV3、SegNet、U-Net++ 等分割网络。
- 如果后续能找到成对的低清/高清图片，再补充严格的图像复原或超分辨率定量实验。
