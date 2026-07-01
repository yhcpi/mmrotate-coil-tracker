"""vis_video_final.py - 用 v6.6 detector 生成最终可视化 3.MP4

输出视频内容:
  - 原视频帧 (按目标分辨率缩放)
  - 顶部黑条: 当前状态 (MAIN/SUB) + sm/pk/ratio
  - ENTER/LEAVE 事件触发时: 屏幕底部横条 (绿色=ENTER, 红色=LEAVE)
  - GT 参考时刻 (黄色竖线 + 标注)

GT (3mp4_visual_states.md):
  C1 LEAVE @ 61.5s
  C2 LEAVE @ 129.5s
  C3 LEAVE @ 191.0s
"""
import sys, cv2, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')
from pyav_reader import open_video
from frame_diff_detector import FrameDiffCoilDetector

GT_LEAVES = [61.5, 129.5, 191.0]
GT_ENTERS = [13.5, 83.0, 148.0, 226.0]


def open_writer(path, fps, w, h):
    base, ext = os.path.splitext(path)
    ext_lower = ext.lower()
    for codec in ('mp4v', 'XVID', 'MJPG'):
        p = path
        if codec != 'mp4v' and ext_lower == '.mp4':
            p = base + '.avi'
        w_ = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if w_.isOpened():
            print(f'输出: {p} ({w}x{h}@{fps:.1f}fps, codec={codec})')
            return w_, p
        w_.release()
    return None, None


def main():
    in_path = '/home/pi/projects/mm/3.mp4'
    out_path = '/home/pi/projects/mm/帧差法/3.MP4'
    # 用 1.0x 原分辨率: 测试过 downscale=0.5 时 C2 LEAVE 触发提前 20s (resolution invariance 失效)
    # 保持 1.0x 保证事件时间一致

    # 第一遍: 跑 detector 收集事件
    cap = open_video(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'输入: {src_w}x{src_h}@{fps:.1f}fps  输出: 1.0x 原分辨率')

    det = FrameDiffCoilDetector(fall_peak_ratio=0.50, unsaturating_ratio=0.85)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        det.update(frame, fps=fps)
    cap.release()
    events = [(fidx / fps, ev) for fidx, ev, _ in det.event_log]
    print(f'检测事件: {[(round(t,1), ev) for t, ev in events]}')

    # 第二遍: 写入视频 (原分辨率)
    cap = open_video(in_path)
    new_w, new_h = src_w, src_h
    writer, real_path = open_writer(out_path, fps, new_w, new_h)
    if writer is None:
        sys.exit(1)

    det = FrameDiffCoilDetector(fall_peak_ratio=0.50, unsaturating_ratio=0.85)
    i = 0
    bar_h = max(50, new_h // 12)
    font_scale_main = max(0.5, new_h / 720)
    print('开始写视频...')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        t = i / fps
        _, state, sm = det.update(frame, fps=fps)
        sub = det.sub_state

        # === 顶部状态条 ===
        cv2.rectangle(frame, (0, 0), (new_w, bar_h), (0, 0, 0), -1)
        main_color = (0, 0, 255) if state == 'CHANGE' else (0, 255, 0)
        cv2.putText(frame, f'MAIN:{state}  SUB:{sub}',
                    (10, int(bar_h * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, main_color, 2)
        peak = det._coil_peak
        ratio = f'{sm/peak:.3f}' if peak > 0 else 'n/a'
        cv2.putText(frame, f'sm={int(sm):>6d} pk={int(peak):>6d} r={ratio}',
                    (10, int(bar_h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main * 0.7, (255, 255, 255), 1)

        # === 时间戳 (右下角) ===
        cv2.putText(frame, f't={t:>5.2f}s F{i}',
                    (new_w - 220, new_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_main * 0.7, (255, 255, 255), 1)

        # === GT 时刻标线 (黄色竖线穿过状态条) ===
        for gt_t in GT_ENTERS + GT_LEAVES:
            gt_frame = int(gt_t * fps)
            if abs(i - gt_frame) < 3:
                cv2.line(frame, (0, bar_h), (new_w, bar_h + 60), (0, 255, 255), 3)
                is_leave = gt_t in GT_LEAVES
                label = f'GT {"LEAVE" if is_leave else "ENTER"} t={gt_t:.1f}s'
                cv2.putText(frame, label, (10, bar_h + 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale_main * 0.8, (0, 255, 255), 2)

        # === 检测事件触发 (整屏闪动) ===
        for ev_idx, (ev_t, ev_type) in enumerate(events):
            ev_frame = int(ev_t * fps)
            delta = i - ev_frame
            if 0 <= delta < 10:  # 触发后 10 帧内显示
                color = (0, 255, 0) if ev_type == 'ENTER' else (0, 0, 255)
                # 底部横条
                cv2.rectangle(frame, (0, new_h - 80), (new_w, new_h), color, -1)
                # bias 只对 LEAVE 算且只算前 3 次 (C1/C2/C3), ENTER/第4 coil 不算
                if ev_type == 'LEAVE' and ev_idx // 2 < len(GT_LEAVES):
                    bias = ev_t - GT_LEAVES[ev_idx // 2]
                    txt = f'DETECTED {ev_type} t={ev_t:.2f}s  bias={bias:+.2f}s'
                else:
                    txt = f'DETECTED {ev_type} t={ev_t:.2f}s'
                cv2.putText(frame, txt,
                            (10, new_h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale_main * 0.9, (0, 0, 0), 2)

        writer.write(frame)
    cap.release()
    writer.release()
    if os.path.exists(real_path):
        sz = os.path.getsize(real_path) / 1024 / 1024
        print(f'Done: -> {real_path} ({sz:.1f} MB)')


if __name__ == '__main__':
    main()
