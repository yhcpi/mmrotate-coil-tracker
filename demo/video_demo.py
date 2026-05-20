import argparse
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from mmrotate.core import obb2poly
import mmrotate


def parse_args():
    parser = argparse.ArgumentParser(description='MMRotate 实时视频检测')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='权重文件路径')
    parser.add_argument('--video', default=None, help='视频文件路径（缺省则调用摄像头）')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--score-thr', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--output', default=None, help='保存视频路径（可选）')
    parser.add_argument('--max-display', type=int, default=1280,
                        help='显示窗口的最大宽度（等比例缩放）')
    parser.add_argument('--port', type=int, default=8080, help='网页端口')
    parser.add_argument('--show-fps', action='store_true',
                        help='画面上显示FPS')
    parser.add_argument('--no-web', action='store_true',
                        help='不启动网页流')
    return parser.parse_args()


latest_frame = None
frame_lock = threading.Lock()


class MJPEGHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''<html><head><title>MMRotate 实时检测</title>
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
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def draw_obb(img, poly, score, color=(0, 255, 0), thickness=2):
    pts = poly.reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    label = f'rail {score:.2f}'
    cv2.putText(img, label, (pts[0, 0], pts[0, 1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print('错误: 无法打开视频源')
        print('  - 指定 --video 文件路径')
        print('  - 摄像头默认使用设备 0，可尝试改为 1 等')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'视频源: {w}x{h} @ {fps:.1f}fps, 总帧数: {total if total > 0 else "N/A（摄像头）"}')

    out = None
    if args.output:
        sw = w if w <= args.max_display else args.max_display
        sh = int(h * sw / w)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (sw, sh))
        print(f'输出视频: {args.output} ({sw}x{sh})')

    display_width = args.max_display
    display_height = int(h * display_width / w)

    if not args.no_web:
        server = HTTPServer(('0.0.0.0', args.port), MJPEGHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f'\n🌐 浏览器打开: http://localhost:{args.port}/')
        print('   按 Ctrl+C 停止\n')

    frame_count = 0
    t_start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if total > 0:
                    print(f'视频播放完毕，共 {frame_count} 帧')
                break

            result = inference_detector(model, frame)

            for cls_dets in result:
                if cls_dets is None or len(cls_dets) == 0:
                    continue
                for det in cls_dets:
                    score = float(det[5])
                    if score < args.score_thr:
                        continue
                    det_tensor = torch.from_numpy(det[:5].copy()).float().unsqueeze(0)
                    poly = obb2poly(det_tensor, 'le90')[0].numpy()
                    draw_obb(frame, poly, score)

            if args.show_fps:
                elapsed = time.time() - t_start
                avg_fps = (frame_count + 1) / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f'{avg_fps:.1f} FPS', (10, 30),
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
                print(f'\r处理中: {frame_count} 帧, {frame_count/elapsed:.1f} FPS', end='', flush=True)

    except KeyboardInterrupt:
        print('\n用户中断')

    elapsed = time.time() - t_start
    if frame_count > 0:
        print(f'\n完成: {frame_count} 帧, {elapsed:.1f}s, '
              f'平均 {frame_count/elapsed:.1f} FPS')

    cap.release()
    if out:
        out.release()


if __name__ == '__main__':
    main()
