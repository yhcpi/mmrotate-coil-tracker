# Copyright (c) OpenMMLab. All rights reserved.
"""
Data augmentation visualization for rail detection.

Two modes:
1. PIPELINE VIEW: For each step in train_pipeline, show the result of applying
   ONLY that step to the original image (single shot, no chained transforms).
   Helps the user judge per-step intensity.

2. RANDOM SWEEP: Apply the full train_pipeline N times to the same image
   with different random seeds, to show the range of augmentation outcomes.

Output is written to /home/pi/projects/mm/result_vis_v2/aug_viz/
"""
import argparse
import os
import os.path as osp
import sys
from copy import deepcopy

import cv2
import mmcv
import numpy as np
from mmcv import Config
from numpy import random as np_random


def _ensure_color(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _draw_poly(img, polys, color=(0, 255, 0), thickness=2):
    img = img.copy()
    for poly in polys:
        pts = poly.reshape(-1, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img


def _parse_label_file(label_path):
    """Parse DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class difficult."""
    polys = []
    if not osp.exists(label_path):
        return polys
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            coords = list(map(float, parts[:8]))
            polys.append(np.array(coords))
    return polys


def _resize_to_height(img, h=720):
    scale = h / img.shape[0]
    new_w = int(img.shape[1] * scale)
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)


def _tile_grid(images, labels, cols=3, gap=10, bg_color=(50, 50, 50)):
    """Tile a list of (image, label) pairs into a single grid image."""
    images = [_ensure_color(img) for img in images]
    # Resize all to a common height for fair comparison
    target_h = min(img.shape[0] for img in images)
    images = [_resize_to_height(img, target_h) for img in images]

    h = target_h + 40  # 40px for text strip
    rows = (len(images) + cols - 1) // cols
    cell_w = max(img.shape[1] for img in images)
    grid_w = cols * cell_w + (cols + 1) * gap
    grid_h = rows * h + (rows + 1) * gap
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = i // cols, i % cols
        x = gap + c * (cell_w + gap)
        y = gap + r * (h + gap)
        # Paste image (left-aligned in cell)
        grid[y:y + img.shape[0], x:x + img.shape[1]] = img
        # Draw label text
        cv2.putText(grid, label, (x, y + img.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return grid


def _make_base_results(img):
    """Build a minimum results dict that satisfies most pipeline transforms."""
    return {
        'img': img,
        'img_fields': ['img'],
        'img_shape': img.shape,
        'ori_shape': img.shape,
        'gt_bboxes': np.zeros((0, 5), dtype=np.float32),
        'gt_labels': np.zeros((0,), dtype=np.int64),
        'bbox_fields': [],
        'gt_bboxes_ignore': np.zeros((0, 5), dtype=np.float32),
        'gt_labels_ignore': np.zeros((0,), dtype=np.int64),
    }


def make_step_viz(img, polys, step_cfg, polys_transform=None):
    """Apply a single pipeline step to (img, polys).

    polys_transform: optional callable(polys, transform) -> polys
                     where transform is the dict returned by the pipeline step.
                     For rotations/flips we also need to transform the polygons.
    """
    if step_cfg['type'] in ('LoadImageFromFile', 'LoadAnnotations',
                            'DefaultFormatBundle', 'Collect'):
        return img, polys

    from mmrotate.datasets.builder import ROTATED_PIPELINES
    from mmrotate.datasets.pipelines import GaussianBlur  # ensure registered
    cls = ROTATED_PIPELINES.get(step_cfg['type'])
    if cls is None:
        print(f'  [skip] {step_cfg["type"]}: not in registry')
        return _draw_poly(_ensure_color(img), polys), polys

    # Build instance from cfg, overriding probabilistic params so the
    # effect is always visible in step-by-step mode.
    cfg = {k: v for k, v in step_cfg.items() if k != 'type'}
    step_type = step_cfg['type']
    if step_type == 'GaussianBlur':
        cfg['prob'] = 1.0
    elif step_type == 'PolyRandomRotate':
        cfg['rotate_ratio'] = 1.0
        cfg['allow_negative'] = True
    elif step_type == 'RRandomFlip':
        cfg['flip_ratio'] = 1.0
    elif step_type in ('BrightnessAdjust', 'ContrastAdjust', 'SaturationAdjust',
                       'ColorJitter', 'Sharpness', 'ColorTemperature'):
        cfg['prob'] = 1.0

    try:
        transform = cls(**cfg)
    except Exception as e:
        print(f'  [skip] {step_cfg["type"]}: cannot construct ({e})')
        return _draw_poly(_ensure_color(img), polys), polys

    # Prepare a fresh results dict with all keys that transforms may need.
    results = _make_base_results(img)

    # --- HACK: force flip direction so output is deterministic ----------
    # RRandomFlip (via mmdet RandomFlip) checks `if 'flip' not in results`
    # before consulting flip_ratio.  Pre-set flip=True so the override
    # takes effect regardless of internal rng.
    if step_type == 'RRandomFlip':
        results['flip'] = True
        results['flip_direction'] = 'horizontal'

    out = transform(deepcopy(results))

    # --- defensive: transform may return None or omit 'img' -------------
    if out is None or not isinstance(out, dict) or 'img' not in out:
        out_img = _ensure_color(img)
    else:
        out_img = out['img']
        # Some steps (e.g. Normalize) return float32 — cv2 needs uint8.
        if out_img.dtype != np.uint8:
            out_img = out_img.astype(np.uint8)

    # Re-apply polygon overlay on the output image.
    if polys is not None and polys_transform is not None:
        try:
            transformed_polys = polys_transform(polys, transform)
            out_img = _draw_poly(out_img, transformed_polys)
        except Exception:
            out_img = _draw_poly(out_img, polys)  # fall back to original polys
    elif polys is not None:
        out_img = _draw_poly(out_img, polys)
    return out_img, polys


def poly_rotate(polys, transform):
    """Best-effort: ignore rotation for now (we don't have the exact matrix)."""
    return polys


def poly_flip_h(polys, transform):
    """For visualization we don't actually flip the polygon since we don't
    know img_shape here. Just return original."""
    return polys


def _resize_max_side(img, max_side=1600):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _prepare_save(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return _resize_max_side(img)


def viz_pipeline_steps(img, polys, train_pipeline, name, out_dir):
    probabilistic_steps = {'PolyRandomRotate', 'RRandomFlip',
                          'PhotoMetricDistortion', 'GaussianBlur',
                          'BrightnessAdjust', 'ContrastAdjust',
                          'SaturationAdjust', 'ColorJitter',
                          'Sharpness', 'ColorTemperature'}
    prefix = name.replace('.png', '')

    def _write(out_img, suffix):
        out_img = _prepare_save(out_img)
        fname = f'{prefix}__{suffix}.png'
        path = osp.join(out_dir, fname)
        cv2.imwrite(path, out_img)
        print(f'  wrote {path}')

    _write(_draw_poly(_ensure_color(img), polys), '00_original')

    step_idx = 1
    for i, step in enumerate(train_pipeline):
        step_type = step['type']
        if step_type in ('LoadImageFromFile', 'LoadAnnotations',
                         'DefaultFormatBundle', 'Collect'):
            continue

        if step_type in probabilistic_steps:
            for k in range(4):
                np_random.seed(42 + i * 100 + k)
                out_img, _ = make_step_viz(img, polys, step)
                _write(out_img, f'{step_idx:02d}_{step_type}_s{k}')
        else:
            out_img, _ = make_step_viz(img, polys, step)
            _write(out_img, f'{step_idx:02d}_{step_type}')

        step_idx += 1


def viz_random_sweep(img, polys, train_pipeline, n_samples, seed_base, name, out_dir):
    from mmrotate.datasets.builder import ROTATED_PIPELINES
    from mmrotate.datasets.pipelines import GaussianBlur  # noqa: F401

    from mmdet.datasets.pipelines import Compose
    steps = []
    for s in train_pipeline:
        if s['type'] in ('LoadImageFromFile', 'LoadAnnotations',
                         'DefaultFormatBundle', 'Collect'):
            continue
        cfg = {k: v for k, v in s.items() if k != 'type'}
        if s['type'] == 'PolyRandomRotate':
            cfg['allow_negative'] = True
        cls = ROTATED_PIPELINES.get(s['type'])
        if cls is None:
            print(f'  [skip] {s["type"]}: not in registry')
            continue
        try:
            steps.append(cls(**cfg))
        except Exception as e:
            print(f'  [skip] {s["type"]}: {e}')
    pipeline = Compose(steps)

    prefix = name.replace('.png', '')
    for k in range(n_samples):
        np_random.seed(seed_base + k)
        results = _make_base_results(img.copy())
        out = pipeline(results)

        if out is None or not isinstance(out, dict) or 'img' not in out:
            out_img = _ensure_color(img)
        else:
            out_img = out['img']
            if out_img.dtype != np.uint8:
                out_img = out_img.astype(np.uint8)
        out_img = _prepare_save(_draw_poly(out_img, polys))

        fname = f'{prefix}__full_s{k}.png'
        path = osp.join(out_dir, fname)
        cv2.imwrite(path, out_img)
        print(f'  wrote {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/rotated_fcos/rotated_fcos_r18_fpn_1x_rail_kld_le90.py')
    parser.add_argument('--data-root', default='data/')
    parser.add_argument('--n-samples', type=int, default=6, help='random sweep count')
    parser.add_argument('--n-images', type=int, default=2, help='number of source images')
    parser.add_argument('--out-dir', default='result_vis_v2/aug_viz')
    parser.add_argument('--no-sweep', action='store_true', help='skip full pipeline random sweep')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    train_pipeline = cfg.train_pipeline

    img_dir = osp.join(args.data_root, 'train/images')
    label_dir = osp.join(args.data_root, 'train/labels')
    img_names = sorted(os.listdir(img_dir))[:args.n_images]

    for n, name in enumerate(img_names):
        img_path = osp.join(img_dir, name)
        label_path = osp.join(label_dir, name.replace('.png', '.txt').replace('.jpg', '.txt'))
        polys = _parse_label_file(label_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f'  skip {name} (cannot read)')
            continue
        print(f'\n=== image {n+1}/{len(img_names)}: {name} ({img.shape})  polys={len(polys)} ===')

        viz_pipeline_steps(img, polys, train_pipeline, name, args.out_dir)

        if not args.no_sweep:
            viz_random_sweep(img, polys, train_pipeline, args.n_samples,
                             seed_base=1000 + n * 100, name=name, out_dir=args.out_dir)

    print(f'\nAll visualizations written to {args.out_dir}/')


if __name__ == '__main__':
    main()
