# Image restoration inputs

Put 2-3 images here if you want the restoration experiment to use real images.

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

After adding images, rerun:

```bash
python run_road_experiments.py --seg-dataset camvid_tiny --epochs 25 --train-count 80 --val-count 20 --restore-train-count 96 --restore-val-count 12 --batch-size 8 --size 96 --output-dir outputs_camvid
```

If this folder has no image files, the script uses three CamVid road-scene images when CamVid is available.
