"""validate_v7.py - 验证 v7 consec 信号 LEAVE 偏差"""
import sys
import cv2
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.')
from pyav_reader import open_video
from frame_diff_detector import FrameDiffCoilDetector


GT_LEAVES = [61.5, 129.5, 191.0]
GT_ENTERS = [13.5, 83.0, 148.0, 226.0]


def main():
    cap = open_video('/home/pi/projects/mm/3.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
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
    print('v7 事件:')
    for t, ev in events:
        print(f'  t={t:>6.2f}s {ev}')

    # 计算 LEAVE bias
    leaves = [t for t, ev in events if ev == 'LEAVE']
    if len(leaves) >= 3:
        biases = [leaves[i] - GT_LEAVES[i] for i in range(3)]
        print(f'\nLEAVE bias (前 3 个): {biases}')
        print(f'max abs bias: {max(abs(b) for b in biases):.2f}s')
        target = 0.5
        if max(abs(b) for b in biases) <= target:
            print(f'✓ 全部 <{target}s')
        else:
            print(f'✗ 超过 {target}s')


if __name__ == '__main__':
    main()