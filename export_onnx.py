"""Export RotatedFCOS model to ONNX (runs on OLD machine with mmrotate env).

Output: 15 tensors = 5 FPN levels × (cls_score, bbox_pred(5ch), centerness)
Post-processing (sigmoid, top-k, NMS, decode) is done in Python.
"""
import argparse
import os

import torch
from mmdet.apis import init_detector
import mmrotate  # noqa: F401  ← 触发 RotatedFCOS 注册


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='配置文件')
    p.add_argument('checkpoint', help='权重文件')
    p.add_argument('--output', default='rail.onnx', help='ONNX 输出路径')
    p.add_argument('--input-size', type=int, nargs=2, default=[800, 800],
                   metavar=('H', 'W'), help='导出输入尺寸 (动态轴仍可用)')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


class ExportWrapper(torch.nn.Module):
    """Wrap backbone+neck+head. Output: tuple of 15 tensors (5 levels × 3)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.backbone(x)
        if self.model.with_neck:
            feats = self.model.neck(feats)
        # head returns 4 lists: cls_scores, bbox_preds(4ch), angle_preds(1ch), centernesses
        cls_scores, bbox_preds, angle_preds, centernesses = self.model.bbox_head(feats)
        out = []
        for i in range(len(cls_scores)):
            out.append(cls_scores[i])
            out.append(bbox_preds[i])
            out.append(angle_preds[i])
            out.append(centernesses[i])
        return tuple(out)


def main():
    args = parse_args()
    H, W = args.input_size
    print(f'加载模型: {args.config}')
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    wrapped = ExportWrapper(model).to(args.device).eval()

    # 试跑一次
    dummy = torch.randn(1, 3, H, W, device=args.device)
    with torch.no_grad():
        outs = wrapped(dummy)
    print(f'  输出: {len(outs)} 个 tensor')
    for i, o in enumerate(outs):
        kind = ['cls', 'bbox', 'angle', 'cent'][i % 4]
        lvl = i // 4
        print(f'    [{i}] level{lvl}/{kind}: {o.shape}')

    # 导 ONNX
    print(f'\n导 ONNX 到: {args.output}')
    torch.onnx.export(
        wrapped,
        dummy,
        args.output,
        input_names=['input'],
        output_names=[f'out_{i}' for i in range(len(outs))],
        dynamic_axes={
            'input': {0: 'N', 2: 'H', 3: 'W'},
            **{f'out_{i}': {0: 'N'} for i in range(len(outs))},
        },
        opset_version=13,
        do_constant_folding=True,
    )

    sz = os.path.getsize(args.output) / 1024 / 1024
    print(f'✓ 导出完成: {args.output}  ({sz:.1f} MB)')


if __name__ == '__main__':
    main()
