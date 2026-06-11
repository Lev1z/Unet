# Image restoration inputs

Put 2-3 images here if you want the restoration experiment to use real images.

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

After adding images, rerun:

```bash
python run_road_experiments.py --epochs 25 --train-count 768 --val-count 192 --restore-train-count 96 --restore-val-count 12 --batch-size 32 --size 96 --output-dir outputs_road
```

If this folder has no image files, the script uses three generated road-scene images as placeholders.
