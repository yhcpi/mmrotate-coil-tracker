"""pyav_reader.py - pyav VideoCapture 兼容接口 (海康 IMKH 也能读)

替代 cv2.VideoCapture, 提供相同的 read()/get() 接口, 用于海康 IMKH 等 cv2 不识别的格式.

用法:
    cap = PyAvCapture('/path/to/video.mp4')
    while True:
        ret, frame = cap.read()  # frame: BGR (跟 cv2 一致)
        if not ret: break
"""
import av
import numpy as np


class PyAvCapture:
    def __init__(self, path):
        self._container = av.open(path)
        self._stream = self._container.streams.video[0]
        self._fps = float(self._stream.average_rate) if self._stream.average_rate else 25.0
        # 帧总数: 用 duration * fps 估算 (容器里 frames 经常为 0)
        if self._stream.frames and self._stream.frames > 0:
            self._total = self._stream.frames
        elif self._stream.duration and self._stream.time_base:
            self._total = int(float(self._stream.duration * self._stream.time_base) * self._fps)
        else:
            self._total = 0
        self._w = self._stream.codec_context.width
        self._h = self._stream.codec_context.height
        self._closed = False
        # 已解码帧缓存 (pyav 是迭代器, 不能 rewind)
        self._decoded = []  # list of (pts, frame_bgr)
        self._pos = 0

    def get(self, prop):
        # 兼容 cv2 CAP_PROP_* 常量
        from cv2 import CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
        if prop == CAP_PROP_FPS:
            return self._fps
        elif prop == CAP_PROP_FRAME_COUNT:
            return self._total
        elif prop == CAP_PROP_FRAME_WIDTH:
            return self._w
        elif prop == CAP_PROP_FRAME_HEIGHT:
            return self._h
        return 0

    def read(self):
        # 先返回缓存里的
        if self._pos < len(self._decoded):
            f = self._decoded[self._pos]
            self._pos += 1
            return True, f

        # 解码下一帧
        if self._closed:
            return False, None
        try:
            frame = next(self._container.decode(video=0))
        except (StopIteration, av.AVError):
            self._closed = True
            return False, None
        img = frame.to_ndarray(format='bgr24')
        self._decoded.append(img)
        self._pos += 1
        return True, img

    def release(self):
        if not self._closed:
            self._container.close()
            self._closed = True

    def isOpened(self):
        return not self._closed


def open_video(path):
    """优先 cv2, 失败时回退 pyav (支持海康 IMKH 等格式)"""
    import cv2
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        # 测试能不能真的读到一帧
        ret, frame = cap.read()
        if ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
            return cap
        cap.release()
    # 回退 pyav
    print(f'[INFO] cv2 读不了 {path}, 改用 pyav')
    return PyAvCapture(path)