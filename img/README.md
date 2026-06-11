# 清晰度增强输入

这里放 2-3 张用于清晰度增强展示的图片。当前主实验直接读取本目录下的 `1.jpg`、`2.jpg`、`3.jpg`。

支持格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`。

添加或替换图片后，在仓库根目录重新运行：

```bash
python run_road_experiments.py --seg-dataset camvid_tiny --epochs 25 --train-count 80 --val-count 20 --batch-size 8 --size 96 --output-dir outputs_camvid
```

如果本目录没有图片，脚本会在 CamVid 可用时使用三张 CamVid 道路场景图作为备用输入。

注意：当前清晰度增强是无参考视觉展示，不需要高清真值图，也不计算 MSE、PSNR、SSIM。
