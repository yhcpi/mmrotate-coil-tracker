import argparse
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.apis import init_detector
from mmrotate.core import obb2poly
import mmrotate


def parse_args():
    parser = argparse.ArgumentParser(description='MMRotate 钢轨+钢卷检测')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='权重文件路径')
    parser.add_argument('--video', default=None, help='视频文件路径（缺省则调用摄像头）')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--score-thr', type=float, default=0.3, help='钢轨置信度阈值')
    parser.add_argument('--output', default=None, help='保存视频路径（可选）')
    parser.add_argument('--max-display', type=int, default=1280,
                        help='显示窗口的最大宽度（等比例缩放）')
    parser.add_argument('--port', type=int, default=8080, help='网页端口')
    parser.add_argument('--show-fps', action='store_true', help='画面上显示FPS')
    parser.add_argument('--no-web', action='store_true', help='不启动网页流')
    parser.add_argument('--skip-frame', type=int, default=1,
                        help='跳帧数: 每skip_frame帧做一次检测(默认1=每帧检测)')
    parser.add_argument('--infer-size', type=int, default=None,
                        help='推理分辨率(短边), 缩小后送入模型提升速度')
    return parser.parse_args()


def build_fast_inference(model, infer_size=None):
    """构建 GPU 加速推理函数.
    
    CPU: 仅 cv2.resize (0.4ms)
    GPU: BGR→RGB + normalize (0.8ms vs CPU 43ms) + pad + forward
    相比 inference_detector (每次重建pipeline + CPU normalize) 快 3~5 倍
    """
    device = next(model.parameters()).device
    cfg = model.cfg
    norm = cfg.img_norm_cfg
    mean = torch.tensor(norm['mean'], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm['std'], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    to_rgb = norm.get('to_rgb', True)

    # 从 pipeline 配置读取推理分辨率
    target_w, target_h = cfg.data.test.pipeline[1]['img_scale']

    def infer(frame):
        h, w = frame.shape[:2]
        if infer_size:
            scale = infer_size / min(h, w)
            if scale < 1:
                iw, ih = int(w * scale), int(h * scale)
            else:
                iw, ih = w, h
        else:
            iw, ih = target_w, target_h

        sx, sy = iw / w, ih / h  # 用于 rescale

        # CPU resize 0.4ms（唯一在 CPU 上的操作）
        if iw != w or ih != h:
            resized = cv2.resize(frame, (iw, ih))
        else:
            resized = frame

        # GPU: BGR→RGB + normalize (0.8ms, vs CPU 43ms)
        tensor = torch.from_numpy(resized).to(device, non_blocking=True).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        if to_rgb:
            tensor = tensor.flip(1)  # BGR→RGB
        tensor = (tensor - mean) / std

        # GPU pad to 32x divisor
        ph = (32 - tensor.shape[2] % 32) % 32
        pw = (32 - tensor.shape[3] % 32) % 32
        if ph or pw:
            tensor = F.pad(tensor, (0, pw, 0, ph))

        img_metas = [{
            'filename': None,
            'ori_filename': None,
            'ori_shape': (h, w, 3),
            'img_shape': (ih, iw, 3),
            'pad_shape': (ih + ph, iw + pw, 3),
            'scale_factor': np.array([sx, sy, sx, sy], dtype=np.float32),
            'flip': False,
            'flip_direction': None,
            'img_norm_cfg': norm,
        }]

        with torch.no_grad():
            results = model(return_loss=False, rescale=True, img=[tensor], img_metas=[img_metas])

        return results[0]

    return infer


latest_frame = None
frame_lock = threading.Lock()


class MJPEGHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''<html><head><title>MMRotate 钢轨+钢卷检测</title>
<style>body{margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img src="/stream" id="img">
<script>setInterval(function(){document.getElementById("img").src="/stream?"+new Date().getTime()},50)</script>
</body></html>'''
            self.wfile.write(html.encode())
        elif self.path.startswith('/stream'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        if latest_frame is None:
                            time.sleep(0.01)
                            continue
                        buf = latest_frame
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(buf)}\r\n\r\n'.encode())
                    self.wfile.write(buf)
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class RailHistoryTracker:
    def __init__(self, max_history=30):
        self.max_history = max_history
        self.history = []
        self.last_rail_info = None

    def update(self, rail_info):
        if rail_info is not None:
            self.history.append(rail_info)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            self.last_rail_info = self._smoothed_info()

    def _smoothed_info(self):
        if not self.history:
            return None
        dirs = [h.get('direction') for h in self.history if h]
        if not dirs:
            return None
        majority_dir = max(set(dirs), key=dirs.count)
        same_dir = [h for h in self.history if h.get('direction') == majority_dir]
        if not same_dir:
            return self.history[-1]
        smoothed = {'direction': majority_dir}
        for k in same_dir[0]:
            if k == 'direction':
                continue
            vals = [h[k] for h in same_dir if k in h]
            smoothed[k] = sum(vals) / len(vals)
        return smoothed

    def get_last_rail_info(self):
        return self.last_rail_info

    def has_history(self):
        return self.last_rail_info is not None


rail_tracker = RailHistoryTracker(max_history=30)


def process_rails_and_coils(rail_polys, im_h, im_w):
    if len(rail_polys) == 0:
        if rail_tracker.has_history():
            last_info = rail_tracker.get_last_rail_info()
            return _coils_from_history(last_info, im_h, im_w)
        else:
            return [[0, 0, im_w, 0, im_w, im_h, 0, im_h]]

    all_pts = rail_polys.reshape(-1, 2)
    xs, ys = all_pts[:, 0], all_pts[:, 1]
    min_x, min_y = xs.min(), ys.min()
    max_x, max_y = xs.max(), ys.max()

    rail_w = max_x - min_x
    rail_h = max_y - min_y
    is_horizontal = rail_w >= rail_h

    if is_horizontal:
        if rail_w / im_w > 0.9:
            return []
    else:
        if rail_h / im_h > 0.9:
            return []

    current_info = {
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'rail_w': rail_w, 'rail_h': rail_h,
        'direction': 'horizontal' if is_horizontal else 'vertical',
    }

    if is_horizontal:
        area_top = rail_w * min_y
        area_bottom = rail_w * (im_h - max_y)
        if area_top >= area_bottom:
            box = [min_x, 0, max_x, 0, max_x, min_y, min_x, min_y]
        else:
            box = [min_x, max_y, max_x, max_y, max_x, im_h, min_x, im_h]
        current_info['rail_x_min'] = min_x
        current_info['rail_x_max'] = max_x
    else:
        area_left = min_x * rail_h
        area_right = (im_w - max_x) * rail_h
        if area_left >= area_right:
            box = [0, min_y, min_x, min_y, min_x, max_y, 0, max_y]
        else:
            box = [max_x, min_y, im_w, min_y, im_w, max_y, max_x, max_y]
        current_info['rail_y_min'] = min_y
        current_info['rail_y_max'] = max_y

    rail_tracker.update(current_info)
    return [box]


def _coils_from_history(info, im_h, im_w):
    if info['direction'] == 'horizontal':
        rx0, rx1 = info['rail_x_min'], info['rail_x_max']
        my0, my1 = info['min_y'], info['max_y']
        if (rx1 - rx0) * my0 >= (rx1 - rx0) * (im_h - my1):
            return [[rx0, 0, rx1, 0, rx1, my0, rx0, my0]]
        else:
            return [[rx0, my1, rx1, my1, rx1, im_h, rx0, im_h]]
    else:
        ry0, ry1 = info['rail_y_min'], info['rail_y_max']
        mx0, mx1 = info['min_x'], info['max_x']
        if mx0 * (ry1 - ry0) >= (im_w - mx1) * (ry1 - ry0):
            return [[0, ry0, mx0, ry0, mx0, ry1, 0, ry1]]
        else:
            return [[mx1, ry0, im_w, ry0, im_w, ry1, mx1, ry1]]


def draw_poly(img, pts, color, thickness, label=None):
    pts = np.array(pts, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        cv2.putText(img, label, (pts[0, 0], pts[0, 1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    # 构建 GPU 加速推理函数，替代每次重新构建 pipeline 的 inference_detector
    fast_infer = build_fast_inference(model, infer_size=args.infer_size)

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print('错误: 无法打开视频源')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'视频源: {w}x{h} @ {fps:.1f}fps')

    out = None
    if args.output:
        sw = w if w <= args.max_display else args.max_display
        sh = int(h * sw / w)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (sw, sh))
        print(f'输出: {args.output} ({sw}x{sh})')

    display_width = args.max_display
    display_height = int(h * display_width / w)

    if not args.no_web:
        server = HTTPServer(('0.0.0.0', args.port), MJPEGHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f'\n🌐 浏览器: http://localhost:{args.port}/  按 Ctrl+C 停止\n')

    frame_count = 0
    t_start = time.time()
    last_result = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rail_polys = []
            do_infer = (frame_count % args.skip_frame == 0)
            if do_infer:
                result = fast_infer(frame)
                last_result = result
            else:
                result = last_result

            # 批量 obb2poly，避免逐个框低效转换
            if result is not None:
                bboxes, scores = [], []
                for cls_dets in result:
                    if cls_dets is None or len(cls_dets) == 0:
                        continue
                    for det in cls_dets:
                        sc = float(det[5])
                        if sc >= args.score_thr:
                            bboxes.append(det[:5])
                            scores.append(sc)
                if bboxes:
                    polys = obb2poly(
                        torch.from_numpy(np.array(bboxes, dtype=np.float32)).cuda(),
                        'le90'
                    ).cpu().numpy()
                    for poly, sc in zip(polys, scores):
                        rail_polys.append(poly)
                        draw_poly(frame, poly, (0, 255, 0), 2, f'rail {sc:.2f}')

            rail_polys = np.array(rail_polys) if rail_polys else np.zeros((0, 8))

            if not args.no_web:
                is_video = total > 0 and total < 100000
                if not is_video:
                    rail_tracker.history.clear()
                    rail_tracker.last_rail_info = None

            coil_boxes = process_rails_and_coils(rail_polys, h, w)

            label = 'Coil Area'
            if len(rail_polys) == 0:
                if rail_tracker.has_history():
                    label = 'Coil (History)'
                else:
                    label = 'Full Coils'
            for box in coil_boxes:
                draw_poly(frame, box, (0, 0, 255), 2, label)

            status = f'Rails: {len(rail_polys)}'
            if rail_tracker.has_history() and len(rail_polys) == 0:
                status += ' (history)'
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if args.show_fps:
                elapsed = time.time() - t_start
                avg_fps = (frame_count + 1) / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f'{avg_fps:.1f} FPS', (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if display_width != w:
                disp = cv2.resize(frame, (display_width, display_height))
            else:
                disp = frame

            if out:
                out.write(disp)

            if not args.no_web:
                _, jpeg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with frame_lock:
                    global latest_frame
                    latest_frame = jpeg.tobytes()

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                print(f'\r处理: {frame_count} 帧, {frame_count/elapsed:.1f} FPS', end='', flush=True)

    except KeyboardInterrupt:
        print('\n用户中断')

    elapsed = time.time() - t_start
    if frame_count > 0:
        print(f'\n完成: {frame_count} 帧, {elapsed:.1f}s, {frame_count/elapsed:.1f} FPS')

    cap.release()
    if out:
        out.release()


if __name__ == '__main__':
    main()
