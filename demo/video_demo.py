import argparse
import base64
import hashlib
import os
import threading
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer

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
    parser.add_argument('--no-roi', action='store_true', help='跳过 ROI 选择，使用全图')
    parser.add_argument('--roi', type=str, default=None, metavar='X,Y,W,H',
                        help='手动指定检测 ROI 区域，格式: x,y,w,h（像素）')
    parser.add_argument('--log', type=str, default=None,
                        help='DBG 日志保存路径（默认 run_YYYYMMDD_HHMMSS.txt，不指定则只在终端输出）')
    parser.add_argument('--skip-frame', type=int, default=1,
                        help='跳帧数: 每skip_frame帧做一次检测(默认1=每帧检测)')
    parser.add_argument('--infer-size', type=int, default=None,
                        help='推理分辨率(短边), 缩小后送入模型提升速度')
    parser.add_argument('--save-coil', nargs='?', const='coil_saves', default=None, metavar='DIR',
                        help='HEAD/TAIL 状态时自动保存钢卷框内图片 (默认目录 coil_saves/, 1fps/状态限速)')
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
frame_seq = [0]
frame_lock = threading.Lock()
frame_cond = threading.Condition(frame_lock)


class MJPEGHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/ws':
            if self.headers.get('Upgrade', '').lower() != 'websocket':
                self.send_response(400)
                self.end_headers()
                return
            key = self.headers.get('Sec-WebSocket-Key', '')
            if not key:
                self.send_response(400)
                self.end_headers()
                return
            accept = base64.b64encode(
                hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()
            ).decode()
            self.send_response(101, 'Switching Protocols')
            self.send_header('Upgrade', 'websocket')
            self.send_header('Connection', 'Upgrade')
            self.send_header('Sec-WebSocket-Accept', accept)
            self.end_headers()
            self.wfile.flush()
            last_sent = -1
            try:
                while True:
                    with frame_cond:
                        frame_cond.wait(timeout=1.0)
                        buf = latest_frame
                        seq = frame_seq[0]
                    if buf is None or seq == last_sent:
                        continue
                    last_sent = seq
                    n = len(buf)
                    if n < 126:
                        self.wfile.write(bytes([0x82, n]))
                    elif n < 65536:
                        self.wfile.write(bytes([0x82, 126, (n >> 8) & 0xFF, n & 0xFF]))
                    else:
                        self.wfile.write(bytes([0x82, 127]) + n.to_bytes(8, 'big'))
                    self.wfile.write(buf)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path == '/' or self.path.startswith('/?'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            html = '''<html><head><title>MMRotate 钢轨+钢卷检测</title>
<style>body{margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh;font-family:sans-serif;color:#888}
canvas{max-width:100%;max-height:100vh;image-rendering:auto}
#status{position:fixed;top:10px;right:10px;font-size:12px;color:#0f0;background:rgba(0,0,0,0.5);padding:4px 8px;border-radius:4px}
#placeholder{font-size:18px;text-align:center}</style></head>
<body>
<div id="placeholder">⏳ 等待首帧 (主循环+模型加载中)...</div>
<canvas id="canvas" style="display:none"></canvas>
<div id="status">FPS: --</div>
<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const placeholder = document.getElementById("placeholder");
const status = document.getElementById("status");
let pendingBitmap = null;
let frameCount = 0, lastFpsTime = Date.now();
let newCount = 0;

const ws = new WebSocket("ws://" + location.host + "/ws");
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  placeholder.textContent = "🟢 WebSocket 已连接，等待帧...";
};

ws.onmessage = async (event) => {
  if (event.data instanceof ArrayBuffer) {
    try {
      pendingBitmap = await createImageBitmap(new Blob([event.data], {type: "image/jpeg"}));
      newCount++;
    } catch (e) {}
  }
};

ws.onclose = () => {
  placeholder.style.display = "";
  placeholder.textContent = "⚠️ WebSocket 断开，3 秒后重连...";
  setTimeout(() => location.reload(), 3000);
};

ws.onerror = () => {};

function render() {
  if (pendingBitmap) {
    if (canvas.width !== pendingBitmap.width || canvas.height !== pendingBitmap.height) {
      canvas.width = pendingBitmap.width;
      canvas.height = pendingBitmap.height;
    }
    ctx.drawImage(pendingBitmap, 0, 0);
    pendingBitmap = null;
    if (placeholder.style.display !== "none") {
      placeholder.style.display = "none";
      canvas.style.display = "";
    }
    frameCount++;
    const now = Date.now();
    if (now - lastFpsTime > 1000) {
      const fps = (frameCount * 1000 / (now - lastFpsTime)).toFixed(1);
      const newFps = (newCount * 1000 / (now - lastFpsTime)).toFixed(1);
      status.textContent = `渲染: ${fps} FPS | 新帧: ${newFps} FPS`;
      frameCount = 0; newCount = 0; lastFpsTime = now;
    }
  }
  requestAnimationFrame(render);
}

requestAnimationFrame(render);
</script>
</body></html>'''
            self.wfile.write(html.encode())
        elif self.path.startswith('/frame'):
            with frame_cond:
                frame_cond.wait(timeout=1.0)
                buf = latest_frame
                fid = frame_seq[0]
            if buf is None:
                self.send_response(503)
                self.send_header('Content-type', 'text/plain')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(b'no frame')
                return
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('X-Frame-Seq', str(fid))
            self.send_header('Content-Length', str(len(buf)))
            self.end_headers()
            self.wfile.write(buf)
            self.wfile.flush()
        elif self.path.startswith('/stream'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            try:
                last_sent_id = -1
                while True:
                    with frame_cond:
                        frame_cond.wait(timeout=1.0)
                        buf = latest_frame
                        frame_id = frame_seq[0] if frame_seq else 0
                    if buf is None or frame_id == last_sent_id:
                        continue
                    last_sent_id = frame_id
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(buf)}\r\n\r\n'.encode())
                    self.wfile.write(buf)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class RailCoilDetector:
    """四态钢卷检测器 — 替换原 RailHistoryTracker + process_rails_and_coils

    四态: CLEAR / HEAD / NO_RAILS / TAIL
      CLEAR:    rail 数 ≥ max_rails  → 无钢卷，记录参考轨范围
      HEAD:     0 < rail < max 且趋势下降 → 钢卷进入，遮挡铁轨
      NO_RAILS: rail ≈ 0 → 钢卷完全覆盖画面
      TAIL:     0 < rail < max 且趋势上升 → 钢卷离开，铁轨重现

    钢卷框 = 参考全轨范围 − 当前可见轨范围（缺口）
    角度 = 全轨角度 EMA 平滑
    方向自适应：以轨框平均角度为参考系（不依赖图像坐标轴）
    """

    def __init__(self, window=5, median_window=5, confirm_k=2):
        self.window = window
        self.median_window = median_window
        self.confirm_k = confirm_k

        self.state = 'CLEAR'
        self._last_n = None
        self._pending_n = None
        self._pending_count = 0
        self._counts = deque(maxlen=window)

        # CLEAR 稳态时记录参考轨范围
        self.max_rails = 0
        self.ref_min = None
        self.ref_max = None
        self.ref_lo = None
        self.ref_hi = None
        self.ref_theta = None
        self.ref_axis_perp = None
        self.ref_axis_along = None

        self.coil_boxes = []
        self.coil_label = ''
        self.rail_count = 0

        self._theta_smooth = None
        self._theta_ema = 0.9

    @staticmethod
    def _poly_center(poly):
        return np.array([poly[0::2].mean(), poly[1::2].mean()])

    @staticmethod
    def _poly_angle(poly):
        return np.arctan2(poly[3] - poly[1], poly[2] - poly[0])

    @staticmethod
    def _obb_to_poly(cx, cy, w, h, theta):
        ct, st = np.cos(theta), np.sin(theta)
        dwx, dwy = ct * w / 2, st * w / 2
        dhx, dhy = -st * h / 2, ct * h / 2
        return [cx - dwx - dhx, cy - dwy - dhy,
                cx + dwx - dhx, cy + dwy - dhy,
                cx + dwx + dhx, cy + dwy + dhy,
                cx - dwx + dhx, cy - dwy + dhy]

    def _update_reference(self, rail_polys, angles):
        n = len(rail_polys)
        old_max = self.max_rails
        self.max_rails = max(self.max_rails, n)
        all_pts = rail_polys.reshape(-1, 2)
        if n >= self.max_rails and n > 0:
            mean_t = np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())
            if self._theta_smooth is None:
                self._theta_smooth = mean_t
            else:
                diff = mean_t - self._theta_smooth
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                self._theta_smooth += (1 - self._theta_ema) * diff
            self.ref_theta = self._theta_smooth
            self.ref_axis_perp = np.array([np.cos(self.ref_theta),
                                           np.sin(self.ref_theta)])
            self.ref_axis_along = np.array([-np.sin(self.ref_theta),
                                            np.cos(self.ref_theta)])
            cur_perp = all_pts @ self.ref_axis_perp
            cur_along = all_pts @ self.ref_axis_along
            self.ref_min = float(cur_perp.min())
            self.ref_max = float(cur_perp.max())
            self.ref_lo = float(cur_along.min())
            self.ref_hi = float(cur_along.max())

    def get_state_info(self):
        return f'{self.state} rails:{self.rail_count}/{self.max_rails}'

    def _build_box(self, along_center, perp_center, along_span, perp_span):
        if self.ref_axis_perp is None or self.ref_axis_along is None:
            return [0, 0, 0, 0, 0, 0, 0, 0]
        px = perp_center * self.ref_axis_perp[0] + along_center * self.ref_axis_along[0]
        py = perp_center * self.ref_axis_perp[1] + along_center * self.ref_axis_along[1]
        return self._obb_to_poly(px, py, perp_span, along_span, self.ref_theta)

    def _compute_coil_box(self, rail_polys, im_h, im_w):
        self.coil_boxes = []
        if self.state in ('CLEAR', 'NO_RAILS'):
            self.coil_label = ''
            return

        if self.ref_min is None or self.ref_theta is None:
            return

        ref_cw = max(self.ref_hi - self.ref_lo, 1.0)
        ref_ch = max(self.ref_max - self.ref_min, 1.0)
        ref_along_c = (self.ref_lo + self.ref_hi) / 2
        eps = 1.0

        n = len(rail_polys)
        if n < 1:
            return

        perp = rail_polys.reshape(-1, 2) @ self.ref_axis_perp
        cur_min, cur_max = float(perp.min()), float(perp.max())
        g1 = cur_min - self.ref_min
        g2 = self.ref_max - cur_max

        if g1 >= eps and g2 >= eps:
            if g1 >= g2:
                c_min, c_max = self.ref_min, cur_min
            else:
                c_min, c_max = cur_max, self.ref_max
        elif g1 >= eps:
            c_min, c_max = self.ref_min, cur_min
        elif g2 >= eps:
            c_min, c_max = cur_max, self.ref_max
        else:
            return

        ch = max(c_max - c_min, eps)
        cy = (c_min + c_max) / 2
        self.coil_boxes = [self._build_box(ref_along_c, cy, ref_cw, ch)]
        self.coil_label = self.state

    def update(self, rail_polys, im_h, im_w):
        n = len(rail_polys)
        self.rail_count = n
        self._counts.append(n)

        if len(self._counts) >= self.median_window:
            recent = list(self._counts)[-self.median_window:]
        else:
            recent = list(self._counts)
        cur_n = sorted(recent)[len(recent) // 2]

        if self._last_n is None:
            self._last_n = cur_n
            self._pending_count = 0
        elif cur_n == self._last_n:
            self._pending_count = 0
        elif cur_n == self._pending_n:
            self._pending_count += 1
            if self._pending_count >= self.confirm_k:
                self._last_n = cur_n
                self._pending_count = 0
        else:
            self._pending_n = cur_n
            self._pending_count = 1

        if n > 0:
            centers = np.array([self._poly_center(p) for p in rail_polys])
            angles = np.array([self._poly_angle(p) for p in rail_polys])
        else:
            centers = np.zeros((0, 2))
            angles = np.array([])

        committed = self._last_n
        if self.max_rails == 0:
            target = 'CLEAR'
        elif committed == 0:
            target = 'NO_RAILS'
        elif committed >= self.max_rails:
            target = 'CLEAR'
        else:
            if self.state == 'NO_RAILS':
                target = 'TAIL'
            elif self.state == 'CLEAR':
                target = 'HEAD'
            else:
                target = self.state
        self.state = target

        if n > 0:
            mean_t = np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())
            if self._theta_smooth is None:
                self._theta_smooth = mean_t
                self.ref_theta = mean_t
            else:
                diff = mean_t - self._theta_smooth
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                self._theta_smooth += (1 - self._theta_ema) * diff
                self.ref_theta = self._theta_smooth
            self.ref_axis_perp = np.array([np.cos(self.ref_theta),
                                           np.sin(self.ref_theta)])
            self.ref_axis_along = np.array([-np.sin(self.ref_theta),
                                            np.cos(self.ref_theta)])

        if self.state == 'CLEAR' and len(self._counts) >= self.window and n > 0:
            self._update_reference(rail_polys, angles)

        self._compute_coil_box(rail_polys, im_h, im_w)
        return self.state, self.coil_boxes, self.coil_label


def draw_poly(img, pts, color, thickness, label=None):
    pts = np.array(pts, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        cv2.putText(img, label, (pts[0, 0], pts[0, 1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def get_obb_corners(cx, cy, w, h, theta):
    """OBB (cx, cy, w, h, theta) → 4 角点 (TR, BR, BL, TL)."""
    ct, st = np.cos(theta), np.sin(theta)
    hw, hh = w / 2.0, h / 2.0
    return [
        (cx + hw * ct - hh * st, cy + hw * st + hh * ct),  # TR
        (cx + hw * ct + hh * st, cy + hw * st - hh * ct),  # BR
        (cx - hw * ct + hh * st, cy - hw * st - hh * ct),  # BL
        (cx - hw * ct - hh * st, cy - hw * st + hh * ct),  # TL
    ]


def select_roi_interactive(cap):
    """读首帧, 拖动画初始 OBB, 可拖角调大小/拖顶○调角度. 返回 (cx, cy, w, h, theta) 或 None."""
    ret, frame = cap.read()
    if not ret:
        print('无法读取首帧')
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    obb = [None]
    state = {'mode': 'idle',
             'start_mouse': None,
             'start_obb': None}
    HANDLE = 8
    ROT_OFFSET = 35

    def get_handles(o):
        c = get_obb_corners(*o)
        cx, cy, _, _, _ = o
        # 顶边中点 = (TR + TL) / 2，外法线方向 = (mid - center)
        mid_x = (c[0][0] + c[3][0]) / 2.0
        mid_y = (c[0][1] + c[3][1]) / 2.0
        dx, dy = mid_x - cx, mid_y - cy
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        rx, ry = mid_x + dx / n * ROT_OFFSET, mid_y + dy / n * ROT_OFFSET
        return {'TR': c[0], 'BR': c[1], 'BL': c[2], 'TL': c[3], 'ROT': (rx, ry),
                'TOP_MID': (mid_x, mid_y)}

    def hit_test(mx, my, o):
        for name, (hx, hy) in get_handles(o).items():
            if name == 'TOP_MID':
                continue
            if abs(mx - hx) <= HANDLE and abs(my - hy) <= HANDLE:
                return name
        return None

    def point_in_obb(px, py, o):
        cx, cy, w, h, theta = o
        ct, st = np.cos(theta), np.sin(theta)
        lx = (px - cx) * ct + (py - cy) * st
        ly = -(px - cx) * st + (py - cy) * ct
        return abs(lx) <= w / 2 and abs(ly) <= h / 2

    def on_mouse(event, x, y, flags, param):
        if obb[0] is None:
            if event == cv2.EVENT_LBUTTONDOWN:
                state['mode'] = 'drawing'
                state['start_mouse'] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and state['mode'] == 'drawing':
                state['cur_mouse'] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and state['mode'] == 'drawing':
                x1, y1 = state['start_mouse']
                x2, y2 = x, y
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    obb[0] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                              float(abs(x2 - x1)), float(abs(y2 - y1)), 0.0)
                state['mode'] = 'idle'
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                h_name = hit_test(x, y, obb[0])
                if h_name:
                    state['mode'] = 'drag_' + h_name
                    state['start_mouse'] = (x, y)
                    state['start_obb'] = obb[0]
                elif point_in_obb(x, y, obb[0]):
                    state['mode'] = 'drag_move'
                    state['start_mouse'] = (x, y)
                    state['start_obb'] = obb[0]
                else:
                    obb[0] = None
                    state['mode'] = 'idle'
            elif event == cv2.EVENT_MOUSEMOVE and state['mode'] != 'idle':
                mode = state['mode']
                if mode == 'drag_move':
                    dx = x - state['start_mouse'][0]
                    dy = y - state['start_mouse'][1]
                    cx0, cy0, w0, h0, t0 = state['start_obb']
                    obb[0] = (cx0 + dx, cy0 + dy, w0, h0, t0)
                elif mode == 'drag_ROT':
                    cx0, cy0, w0, h0, t0 = state['start_obb']
                    h0_handles = get_handles(state['start_obb'])
                    orig_h = h0_handles['ROT']
                    a0 = np.arctan2(orig_h[1] - cy0, orig_h[0] - cx0)
                    a1 = np.arctan2(y - cy0, x - cx0)
                    obb[0] = (cx0, cy0, w0, h0, t0 + (a1 - a0))
                elif mode.startswith('drag_'):
                    h_name = mode[5:]
                    opp = {'TR': 'BL', 'BR': 'TL', 'BL': 'TR', 'TL': 'BR'}
                    idx = {'TR': 0, 'BR': 1, 'BL': 2, 'TL': 3}
                    anchor = get_obb_corners(*state['start_obb'])[idx[opp[h_name]]]
                    cx0, cy0, w0, h0, t0 = state['start_obb']
                    vx, vy = x - anchor[0], y - anchor[1]
                    ct, st = np.cos(t0), np.sin(t0)
                    w_new = abs(vx * ct + vy * st)
                    h_new = abs(-vx * st + vy * ct)
                    obb[0] = (anchor[0] + vx / 2.0, anchor[1] + vy / 2.0,
                              max(w_new, 5.0), max(h_new, 5.0), t0)
            elif event == cv2.EVENT_LBUTTONUP:
                state['mode'] = 'idle'

    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    sw = min(1280, frame.shape[1])
    sh = int(sw * frame.shape[0] / frame.shape[1])
    cv2.resizeWindow('Select ROI', sw, sh)
    cv2.setMouseCallback('Select ROI', on_mouse)
    print('🖱️  拖动=画初始框  拖角=改大小  拖中心=移动  拖顶○=旋转  '
          '回车=确认  r=重置  s=跳过  ESC=取消')

    while True:
        display = frame.copy()
        if obb[0] is None and state['mode'] == 'drawing':
            x1, y1 = state['start_mouse']
            x2, y2 = state.get('cur_mouse', state['start_mouse'])
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif obb[0] is not None:
            o = obb[0]
            corners = get_obb_corners(*o)
            pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            for name, (hx, hy) in [('TL', corners[3]), ('TR', corners[0]),
                                    ('BR', corners[1]), ('BL', corners[2])]:
                cv2.rectangle(display, (int(hx - HANDLE), int(hy - HANDLE)),
                              (int(hx + HANDLE), int(hy + HANDLE)), (0, 200, 255), -1)
            handles = get_handles(o)
            tm = handles['TOP_MID']
            rh = handles['ROT']
            cv2.line(display, (int(tm[0]), int(tm[1])), (int(rh[0]), int(rh[1])), (0, 200, 255), 1)
            cv2.circle(display, (int(rh[0]), int(rh[1])), HANDLE, (0, 100, 255), -1)
        cv2.putText(display, "ENTER=confirm  r=reset  s=skip-full  ESC=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Frame: {frame.shape[1]}x{frame.shape[0]}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if obb[0] is not None:
            cx, cy, w_, h_, th = obb[0]
            cv2.putText(display,
                        f"OBB: ({cx:.0f},{cy:.0f}) {w_:.0f}x{h_:.0f}  θ={np.rad2deg(th):.1f}°",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.imshow('Select ROI', display)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            break
        elif key == ord('r'):
            obb[0] = None
            state['mode'] = 'idle'
        elif key == ord('s'):
            obb[0] = None
            break
        elif key == 27:
            cv2.destroyWindow('Select ROI')
            return None

    cv2.destroyWindow('Select ROI')
    return obb[0]


def main():
    args = parse_args()

    if args.log is None:
        args.log = f'run_{time.strftime("%Y%m%d_%H%M%S")}.txt'
    log_f = open(args.log, 'w', buffering=1)
    print(f'📝 日志: {args.log}')

    coil_save_dir = None
    if args.save_coil is not None:
        coil_save_dir = args.save_coil
        os.makedirs(coil_save_dir, exist_ok=True)
        for st in ('HEAD', 'TAIL'):
            os.makedirs(os.path.join(coil_save_dir, st), exist_ok=True)
        print(f'💾 钢卷框自动保存: {coil_save_dir}/{{HEAD,TAIL}}/  (1fps/状态)')
    last_coil_save = {'HEAD': 0.0, 'TAIL': 0.0}

    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    # 构建 GPU 加速推理函数，替代每次重新构建 pipeline 的 inference_detector
    fast_infer = build_fast_inference(model, infer_size=args.infer_size)
    detector = RailCoilDetector()

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print('错误: 无法打开视频源')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'视频源: {w}x{h} @ {fps:.1f}fps')

    obb_roi = None
    if not args.no_web:
        server = ThreadingHTTPServer(('0.0.0.0', args.port), MJPEGHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f'\n🌐 浏览器: http://localhost:{args.port}/  按 Ctrl+C 停止\n')

    if args.roi:
        try:
            parts = args.roi.split(',')
            if len(parts) == 4:
                x, y, rw, rh = map(int, parts)
                obb_roi = (x + rw / 2.0, y + rh / 2.0, float(rw), float(rh), 0.0)
            elif len(parts) == 5:
                x, y, rw, rh, deg = map(float, parts)
                obb_roi = (x + rw / 2.0, y + rh / 2.0, rw, rh, np.deg2rad(deg))
            else:
                raise ValueError
        except ValueError:
            print('无效的 --roi 格式，应为 x,y,w,h 或 x,y,w,h,θ_deg')
            return
    elif not args.no_roi:
        obb_roi = select_roi_interactive(cap)
        if obb_roi is None:
            print('未选择 ROI，使用全图')

    roi_mask = None
    aabb = None
    aabb_mask = None
    obb_filter = None
    obb_outline_pts = None
    if obb_roi is not None:
        cx, cy, rw, rh, theta = obb_roi
        corners = get_obb_corners(*obb_roi)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        aabb_x = max(0, int(round(min(xs))))
        aabb_y = max(0, int(round(min(ys))))
        aabb_w = min(w, int(round(max(xs) - min(xs))))
        aabb_h = min(h, int(round(max(ys) - min(ys))))
        aabb = (aabb_x, aabb_y, aabb_w, aabb_h)
        aabb_mask = np.ones((aabb_h, aabb_w), dtype=np.uint8) * 255
        aabb_corners = np.array([(c[0] - aabb_x, c[1] - aabb_y) for c in corners],
                                dtype=np.int32)
        cv2.fillPoly(aabb_mask, [aabb_corners], 0)
        ct_o, st_o = np.cos(theta), np.sin(theta)
        obb_filter = (cx, cy, ct_o, st_o, rw / 2.0, rh / 2.0)
        obb_outline_pts = aabb_corners.reshape(-1, 1, 2)
        work_h, work_w = aabb_h, aabb_w
        print(f'OBB ROI: 中心=({cx:.0f},{cy:.0f}) {rw:.0f}x{rh:.0f} θ={np.rad2deg(theta):.1f}°  '
              f'AABB=({aabb_x},{aabb_y}) {aabb_w}x{aabb_h}')
    else:
        work_h, work_w = h, w

    out = None
    display_width = work_w if work_w <= args.max_display else args.max_display
    display_width = display_width & ~1
    display_height = (int(work_h * display_width / work_w)) & ~1
    if args.output:
        for codec in ('mp4v', 'XVID', 'MJPG'):
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*codec),
                                     fps, (display_width, display_height))
            if writer.isOpened():
                out = writer
                ext = os.path.splitext(args.output)[1].lower()
                if codec != 'mp4v' and ext == '.mp4':
                    base = os.path.splitext(args.output)[0]
                    out.release()
                    args.output = base + '.avi'
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*codec),
                                             fps, (display_width, display_height))
                    out = writer
                print(f'输出: {args.output} ({display_width}x{display_height}, codec={codec})')
                break
            writer.release()
        if out is None:
            print(f'⚠️  无法创建视频输出 ({args.output})，已禁用 output')

    frame_count = 0
    t_start = time.time()
    last_result = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if obb_roi is not None:
                aabb_x, aabb_y, aabb_w, aabb_h = aabb
                work_frame = frame[aabb_y:aabb_y + aabb_h, aabb_x:aabb_x + aabb_w].copy()
            else:
                work_frame = frame

            rail_polys = []
            do_infer = (frame_count % args.skip_frame == 0)
            if do_infer:
                result = fast_infer(frame)
                last_result = result
            else:
                result = last_result

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
                if obb_roi is not None and bboxes:
                    cx_o, cy_o, ct_o, st_o, hw_o, hh_o = obb_filter
                    keep_b, keep_s = [], []
                    for bb, sc in zip(bboxes, scores):
                        dx, dy = bb[0] - cx_o, bb[1] - cy_o
                        lx = dx * ct_o + dy * st_o
                        ly = -dx * st_o + dy * ct_o
                        if abs(lx) <= hw_o and abs(ly) <= hh_o:
                            keep_b.append(bb)
                            keep_s.append(sc)
                    bboxes, scores = keep_b, keep_s
                if obb_roi is not None:
                    aabb_x_off, aabb_y_off = aabb[0], aabb[1]
                    bboxes = [[bb[0] - aabb_x_off, bb[1] - aabb_y_off,
                               bb[2], bb[3], bb[4]] for bb in bboxes]
                if bboxes:
                    polys = obb2poly(
                        torch.from_numpy(np.array(bboxes, dtype=np.float32)).cuda(),
                        'le90'
                    ).cpu().numpy()
                    for poly, sc in zip(polys, scores):
                        rail_polys.append(poly)
                        draw_poly(work_frame, poly, (0, 255, 0), 2, f'rail {sc:.2f}')

            rail_polys = np.array(rail_polys) if rail_polys else np.zeros((0, 8))

            state, coil_boxes, coil_label = detector.update(rail_polys, work_h, work_w)

            if state in ('HEAD', 'TAIL') and len(rail_polys) > 0 and detector.ref_min is not None:
                perp = rail_polys.reshape(-1, 2) @ detector.ref_axis_perp
                cmin, cmax = float(perp.min()), float(perp.max())
                g1 = cmin - detector.ref_min
                g2 = detector.ref_max - cmax
                log_f.write(f'DBG|{frame_count}|{state}|rails={detector.rail_count}|gap=[{g1:.0f},{g2:.0f}]|box={len(coil_boxes)}\n')
            else:
                aabb_x_off = aabb_y_off = 0
                if obb_roi is not None:
                    aabb_x_off, aabb_y_off = aabb[0], aabb[1]
                rail_strs = []
                for bb in bboxes:
                    ocx = bb[0] + aabb_x_off
                    ocy = bb[1] + aabb_y_off
                    ow = bb[2]
                    oh = bb[3]
                    otheta = np.rad2deg(bb[4])
                    rail_strs.append(f'({ocx:.0f},{ocy:.0f},{ow:.0f}x{oh:.0f},θ={otheta:.1f}°)')
                rail_info = '|'.join(rail_strs) if rail_strs else 'none'
                obb_info = ''
                if obb_roi is not None:
                    cx_o, cy_o, w_o, h_o, theta_o = obb_roi
                    obb_info = f'|obb=({cx_o:.0f},{cy_o:.0f},{w_o:.0f}x{h_o:.0f},θ={np.rad2deg(theta_o):.1f}°)|aabb=({aabb_x_off},{aabb_y_off},{aabb[2]}x{aabb[3]})'
                log_f.write(f'DBG|{frame_count}|{state}|rails={detector.rail_count}|rails_pos={rail_info}{obb_info}\n')

            # 顶部状态指示条（任何状态都可见）
            state_names = {
                'CLEAR': 'clear', 'HEAD': 'head',
                'NO_RAILS': 'no rails', 'TAIL': 'tail'}
            state_colors = {
                'CLEAR': (0, 180, 0), 'HEAD': (0, 180, 230),
                'NO_RAILS': (0, 0, 230), 'TAIL': (200, 130, 0)}
            bar_color = state_colors.get(state, (100, 100, 100))
            cv2.rectangle(work_frame, (0, 0), (work_w, 4), bar_color, -1)
            label_text = f'{state_names.get(state, state)} rails:{detector.rail_count}/{detector.max_rails}'
            cv2.putText(work_frame, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
            # 右下角也显示状态（更显眼）
            label_w = len(label_text) * 12
            cv2.rectangle(work_frame, (work_w - label_w - 15, 8), (work_w - 5, 32), (0, 0, 0), -1)
            cv2.putText(work_frame, label_text, (work_w - label_w - 10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

            # 画钢卷框（加粗到 3px，先填半透明再画边线）
            for box in coil_boxes:
                pts = np.array(box, dtype=np.int32).reshape(-1, 2)
                overlay = work_frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 200))
                cv2.addWeighted(overlay, 0.25, work_frame, 0.75, 0, work_frame)
                draw_poly(work_frame, box, (0, 0, 255), 3, coil_label)

            if coil_save_dir is not None and state in ('HEAD', 'TAIL') and coil_boxes:
                now = time.time()
                if now - last_coil_save[state] >= 1.0:
                    last_coil_save[state] = now
                    poly_pts = np.array(coil_boxes[0], dtype=np.float32).reshape(-1, 2)
                    x_min, y_min = poly_pts.min(axis=0)
                    x_max, y_max = poly_pts.max(axis=0)
                    aabb_x_off = aabb[0] if obb_roi is not None else 0
                    aabb_y_off = aabb[1] if obb_roi is not None else 0
                    x0, y0 = max(0, int(x_min + aabb_x_off)), max(0, int(y_min + aabb_y_off))
                    x1, y1 = min(w, int(x_max + aabb_x_off)), min(h, int(y_max + aabb_y_off))
                    if x1 > x0 and y1 > y0:
                        crop = frame[y0:y1, x0:x1].copy()
                        fname = f'frame_{frame_count:07d}_{int(now*1000)}.jpg'
                        cv2.imwrite(os.path.join(coil_save_dir, state, fname), crop,
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])

            # 画 OBB ROI 轮廓（AABB 内坐标，需减去 aabb 偏移）
            if obb_roi is not None:
                cv2.polylines(work_frame, [obb_outline_pts], isClosed=True,
                              color=(255, 0, 255), thickness=2)

            if args.show_fps:
                elapsed = time.time() - t_start
                avg_fps = (frame_count + 1) / elapsed if elapsed > 0 else 0
                cv2.putText(work_frame, f'{avg_fps:.1f} FPS', (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if display_width != work_w:
                disp = cv2.resize(work_frame, (display_width, display_height))
            else:
                disp = work_frame

            if out:
                out.write(disp)

            if not args.no_web:
                ok, jpeg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    global latest_frame
                    with frame_cond:
                        latest_frame = jpeg.tobytes()
                        frame_seq[0] += 1
                        frame_cond.notify_all()

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                print(f'\r处理: {frame_count} 帧, {frame_count/elapsed:.1f} FPS', end='', flush=True)

    except KeyboardInterrupt:
        print('\n用户中断')
    finally:
        cap.release()
        if out:
            out.release()
        log_f.close()
        print(f'📝 日志已保存: {args.log}')

    elapsed = time.time() - t_start
    if frame_count > 0:
        print(f'\n完成: {frame_count} 帧, {elapsed:.1f}s, {frame_count/elapsed:.1f} FPS')


if __name__ == '__main__':
    main()
