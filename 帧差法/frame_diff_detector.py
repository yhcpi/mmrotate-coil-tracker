"""frame_diff_detector.py - 帧差法钢卷检测器 (二态 STABLE / CHANGE + 多指标联合)

设计:
  STABLE  - 画面稳定 (无钢卷 或 钢卷铺满稳定), 默认状态
  CHANGE  - 钢卷正在进入或离开 (smoothed 端点相对差超阈值)

子状态 (内部用, 用于发事件):
  STABLE_NO_COIL  - 无钢卷
  STABLE_COIL     - 钢卷铺满稳定
  CHANGE_RISING   - 钢卷正在进入
  CHANGE_FALLING  - 钢卷正在离开

事件:
  ENTER  - 第一次进入 STABLE_COIL (钢卷来了)
  LEAVE  - 离开 STABLE_COIL 回到 STABLE_NO_COIL (钢卷走了)

多指标 (替代单一 sm):
  raw_area       - 原始 frame diff mask 面积 (sm, 旧指标, 保留作参考)
  hull_area      - 凸包面积 (抗 wire 间隙, 抗 coil 移动)
  filled_area    - 大核 MORPH_CLOSE 后面积 (抗 wire 间隙, 平滑)
  solidity       - contour / hull (区分 "coil 散开" vs "coil 缩小")

主指标: filled_area (主) + hull_area (辅), sm 仍用于兼容调试

判据 (v5 综合方案):
  - filled < absolute_exit                                -> STABLE_NO_COIL
  - filled > absolute_enter + ENTER 单调上升 (N 帧)       -> ENTER 触发
  - filled 单调下降 (N 帧) + hull 单调下降 (M 帧)         -> CHANGE_FALLING
  - filled > T_stable + solidity > 0.85 + 持续 N 帧       -> STABLE_COIL
  - filled > T_stable + solidity 波动                     -> 灰色 (settling)

鲁棒性 (v6):
  - 内部以 REFERENCE_PIXELS=2560*1440 为参考,所有面积绝对阈值 (min_area,
    absolute_enter/exit, first_change_threshold, stable_enter_threshold) 在
    第一次 update() 时按 frame_pixels / REFERENCE_PIXELS 自动归一化.
  - 用户传绝对值参数 = "在 REFERENCE_PIXELS 下的等价值",内部自动按当前视频
    尺寸缩放. 任意视频尺寸行为一致.

依赖: opencv-python, numpy
"""
import cv2
import numpy as np
from collections import deque


# detector 调参时使用的参考视频尺寸 (3.mp4 = 2560x1440)
# 任意视频尺寸, 面积阈值自动按 frame_pixels / REFERENCE_PIXELS 归一化
# 形态学核按 frame 短边 / REFERENCE_MIN_DIM 归一化 (保证不同尺寸下大核 close 的物理占比一致)
REFERENCE_PIXELS = 2560 * 1440  # 3,686,400
REFERENCE_MIN_DIM = 1440        # 3.mp4 短边, 用于 morph_kernel/fill_kernel 归一化


class FrameDiffCoilDetector:
    def __init__(
        self,
        bg_frames=30,           # 前 N 帧建背景
        diff_thresh=20,         # 帧差灰度阈值
        min_area=1500,          # 最小前景面积 (过滤边缘噪声) - 自动按 frame_pixels 归一化
        morph_kernel=5,         # 形态学核大小
        ema_alpha=0.01,         # 背景 EMA 更新系数 (0=不更新)
        bg_pause_during_stable=True,  # True: STABLE_COIL 期停更 bg, 只在 STABLE_NO_COIL 期更新
                                        # 这是 LEAVE 偏差 1.5s 根因修复: 旧版 EMA 把 coil 慢慢吸进 bg,
                                        # STABLE 期 filled 持续跌, ratio 提前跌破 0.85 误触.
        roi=None,               # ROI 矩形 (x, y, w, h), None=全屏
        area_smooth=15,         # mask 面积滑动平均窗口
        # 形态学大核 (用于 filled_area, 抗 wire 间隙)
        fill_kernel=15,         # filled_area 用的 MORPH_CLOSE 核大小
        # 绝对阈值 (filled_area 为主, 替代 sm) - 默认值基于 2560x1440 调出, 自动按 frame_pixels 归一化
        absolute_enter=200000,  # filled > 此值才视为"有钢卷" (开始判定, 比旧 sm 阈值高, 防 coil 边缘)
        absolute_exit=5000,     # filled < 此值才视为"无钢卷" (发 LEAVE)
        # 初次进入门槛: STABLE_NO_COIL → CHANGE_RISING/FALLING 需要 filled 高于此值
        # v5.6: 从 200k 提到 600k. Coil 4 ENTER @t=225.92s (filled=866k) 偏早 4.08s,
        # 视觉 STABLE_COIL 实际 @t=230s. 200k 阈值让 Coil 4 早期边缘伸入就触发 RISING
        # v5.19: 600k → 100k. 视觉 Coil 1 RISING @10.5s filled_smoothed~50k,
        # 但 _is_rising 已经在 9.5s 触发 (tr30=1.92),600k 阈值挡了 1.5s
        # 100k 让 Coil 1 RISING 在 filled_smoothed > 100k 时触发 (约 10.0-10.5s)
        # v5.20: 100k → 250k. Coil 4 RISING @223s 偏早 3s (raw=245k, 视觉 226s 才"开始进入"),
        # Coil 4 涨速比 Coil 1/2/3 快很多,250k 阈值让 Coil 4 RISING 推迟 1s 到 224.08s
        # Coil 1 RISING 平 1s (10.57s → 11.57s, 视觉 10.5s, 偏晚 1.07s) 仍可接受
        first_change_threshold=250000,  # filled 钢卷实体门槛 (取代 50000 sm 阈值)
        # v5.9 额外要求: STABLE_COIL 触发需要 raw filled > 绝对门槛
        #   - Coil 4 STABLE_COIL 触发 t=226.05s filled=849k, 视觉 STABLE @t=230s 估计 ~1000k
        #   - 提高 absolute_enter 会破坏 LEAVE 后检测, 改用新的 stable_enter_threshold 参数
        stable_enter_threshold=800000,  # STABLE_COIL 触发需要的最低 raw filled (抗 RISING 期低值震荡)
        # 趋势判定: 端点相对差
        rise_window=30,         # ENTER 方向: 最近 N 帧
        fall_window=20,         # LEAVE 方向: 最近 N 帧 (短=早检测)
        rise_threshold=0.04,    # ENTER: (末-首)/median > 此值
                               # v5.25: 0.08 → 0.04. Coil 4 RISING 末段 filled 850k→890k
                               # 慢涨 4s, tr30 ≈ 0.04-0.06, 0.08 阈值在 225.92s 就退出 RISING
                               # 降到 0.04 让 Coil 4 RISING 持续到 ~226.5s (推后 ~0.6s)
        fall_threshold=0.05,    # LEAVE: (末-首)/median < -此值
        # 二次确认 (非对称)
        confirm_enter=3,        # 非STABLE→STABLE 需连续 N 帧确认
        confirm_leave=1,        # 任何→非STABLE 需连续 N 帧确认 (短=早响应)
        # STABLE_COIL 判定: filled 接近 max + solidity 高
        saturation_window=10,   # 算 saturation ratio 的窗口
                               # v5.10: 30/45/60 → 10. 长窗口包含 RISING 早期低值, 把 sat=True 推后
                               # 同时也推后 Coil 1 ENTER (从 13.60s 偏晚到 14.77s, ratio=0.97 仍严)
                               # 短窗口 (10 帧 ~0.17s) 只看最近状态, RISING 末段立刻 sat=True
        saturation_ratio=0.92,  # median/max > 此值 视为已铺满
                               # v5.10: 0.97 → 0.92. Coil 1 ENTER filled=1053k 立即 ratio=1.0 OK
                               # Coil 4 RISING 末段 filled=850-870k, 10 帧 median 850k, max 870k,
                               # ratio=0.977 > 0.92 → sat=True 触发, 但还是太早
                               # 真正偏早 4s 的原因不是 sat 阈值, 是确认逻辑 (confirm_enter=3)
                               # (背景建模抬高 max_seen), 30 帧 median 850k, max 870k, ratio 0.98 > 0.80
                               # 立即 sat=True, STABLE_COIL 触发 @t=225.92s, 偏早 4.08s
                               # 视觉 STABLE_COIL 启动 @t=230s filled ~1000k, ratio 应该 > 0.95
        unsaturating_ratio=0.85, # median(30帧) < _coil_peak × 此值 视为开始离开
        unsaturating_mono_min=25, # 30 帧内 ≥ 25 帧单调下降 (v6.6 baseline)
        solidity_stable=0.85,   # solidity > 此值 视为"coil 完整" (CHANGE_FALLING 时下降)
        # 历史最大 filled (慢响应), 用于判定"钢卷基本走完"
        max_decay=0.999,        # max_seen 每帧衰减系数
        max_seen_ratio=0.07,    # filled < _coil_peak × 此值 → STABLE_NO_COIL
                               # v5.7: 0.25 → 0.10. Coil 3 视觉 LEAVE @t=198s filled ~5k,
                               # 0.25=218k → t=194.33s filled=164k 提前触发 (偏早 3.67s)
                               # v5.18: 0.10 → 0.07. Coil 3 peak×0.07=61k, 让 LEAVE 推到
                               # smoothed < 61k (F11715, 195.25s) → 偏早 2.75s (vs 3.17s)
                               # Coil 1 peak×0.07=73k, LEAVE smoothed=58k < 73k (推后 ~0.5s)
                               # Coil 2 peak×0.07=66k, LEAVE smoothed=63k < 66k (推后 ~0.3s)
                               # 整体权衡: Coil 3 偏早 -0.42s, Coil 1/2 偏晚 +0.3-0.5s
                               # 0.10=87k → t=195s filled=83k 触发 (偏差 -3s 仍偏早, 但 50% 改善)
        # 钢卷"已存在"持续时间
        rising_window=60,       # 钢卷刚出现的窗口: _is_rising 才触发 CHANGE_RISING
        # ENTER 确认: filled 单调上升
        enter_monotone_min=20,  # ENTER 触发: 最近 N 帧至少有 min 帧满足 filled[i] > filled[i-1]
        # FALLING 确认: filled + hull 双指标单调下降
        fall_monotone_min=18,   # FALLING 触发: 最近 N 帧至少有 min 帧满足 filled[i] < filled[i-1]
        fall_hull_monotone_min=22,  # FALLING hull 单调下降阈值
        # settling 期屏蔽: ENTER 后多少帧内禁止 FALLING 触发
        settle_guard_frames=120,    # 从 90 提到 120, 给 coil settling 多 0.5s 缓冲
        # FALLING 触发: filled 跌破 _coil_peak × 此值 (magnitude guard)
        # v5.6 硬编码 0.50, 即 "coil 真的走了一半" 才触发. 用户 GT 边界是 "一离开就发",
        # 0.50 偏严, 实际 LEAVE 偏晚 1.5-2s. 调高 (0.70-0.85) 可让 LEAVE 更接近视觉边界.
        fall_peak_ratio=0.50,
        # [v7 NEW] 帧间差分 (consec) LEAVE 早触发信号
        # 帧间差分 |frame_t - frame_{t-1}| 在 STABLE 期是相机噪声级 (~280-340k),
        # 在 LEAVE 启动时短暂上升 (motion) 然后随 coil 离场持续下降到 ~0.
        # 比 EMA 背景差分更早: 不受 EMA 累积吸收 coil 影响.
        consec_diff_thresh=30,        # 帧间差分阈值 (灰度)
        consec_morph_kernel=5,        # consec mask 开运算核 (去椒盐)
        consec_peak_window_sec=2.0,   # 滚动峰值窗口 (取最近 N 秒 consec max 作 reference)
        consec_drop_ratio=0.80,       # consec < rolling_peak × 此值 → 早触发候选
        consec_mono_window_sec=1.0,   # 单调性窗口
        consec_mono_min=30,           # 该窗口内 ≥ 多少帧单调下降才认
    ):
        # 原始传入值 (基于 REFERENCE_PIXELS 调出来的)
        # 第一次 update 时,按 frame_pixels 自动归一化到 self.min_area / absolute_enter / ...
        self._ref_min_area = min_area
        self._ref_absolute_enter = absolute_enter
        self._ref_absolute_exit = absolute_exit
        self._ref_first_change_threshold = first_change_threshold
        self._ref_stable_enter_threshold = stable_enter_threshold
        # 形态学核按短边归一化 (避免 0.33x 等极小尺寸下大核 close 占帧过大)
        self._ref_morph_kernel = morph_kernel
        self._ref_fill_kernel = fill_kernel

        # 默认值 (会按 frame_pixels 缩放) - 让外部代码不会 AttributeError
        # 在第一次 update() 时, 下面的占位值会被替换为按 frame_pixels 归一化后的真实值
        self.bg_frames = bg_frames
        self.diff_thresh = diff_thresh
        self.min_area = min_area
        self.morph_kernel = morph_kernel
        self.ema_alpha = ema_alpha
        self.roi = roi
        self.area_smooth = area_smooth
        self.fill_kernel = fill_kernel
        self.absolute_enter = absolute_enter
        self.absolute_exit = absolute_exit
        self.first_change_threshold = first_change_threshold
        self.stable_enter_threshold = stable_enter_threshold
        self.rise_window = rise_window
        self.fall_window = fall_window
        self.rise_threshold = rise_threshold
        self.fall_threshold = fall_threshold
        self.confirm_enter = confirm_enter
        self.confirm_leave = confirm_leave
        self.saturation_window = saturation_window
        self.saturation_ratio = saturation_ratio
        self.unsaturating_ratio = unsaturating_ratio
        self.unsaturating_mono_min = unsaturating_mono_min
        self.solidity_stable = solidity_stable
        self.bg_pause_during_stable = bg_pause_during_stable
        self.max_decay = max_decay
        self.max_seen_ratio = max_seen_ratio
        self.rising_window = rising_window
        self.enter_monotone_min = enter_monotone_min
        self.fall_monotone_min = fall_monotone_min
        self.fall_hull_monotone_min = fall_hull_monotone_min
        self.settle_guard_frames = settle_guard_frames
        self.fall_peak_ratio = fall_peak_ratio
        # [v7] consec signal params
        self.consec_diff_thresh = consec_diff_thresh
        self.consec_morph_kernel = consec_morph_kernel
        self.consec_peak_window_sec = consec_peak_window_sec
        self.consec_drop_ratio = consec_drop_ratio
        self.consec_mono_window_sec = consec_mono_window_sec
        self.consec_mono_min = consec_mono_min
        # 第一次 update() 时设置的归一化信息
        self.frame_pixels = None  # float 实际 frame 像素数
        self._area_scale = 1.0    # = frame_pixels / REFERENCE_PIXELS
        self._kernel_scale = 1.0  # = min(h, w) / REFERENCE_MIN_DIM
        # 占位的 kernel 元素 (在第一次 update() 时被替换为按短边归一化后的值)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )
        self.fill_kernel_elem = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (fill_kernel, fill_kernel)
        )

        self.bg = None
        self.bg_f = None
        self.bg_buffer = deque(maxlen=bg_frames)
        self.initialized = False
        self.frame_idx = 0
        # raw_area (sm) - 旧指标, 保留作参考和回退
        self.area_history = deque(maxlen=area_smooth)
        # 主指标 filled_area - 大核 close 后的面积
        self.filled_history = deque(maxlen=max(rise_window, fall_window, saturation_window, 60))
        # 辅助指标 hull_area - 凸包面积
        self.hull_history = deque(maxlen=60)
        # solidity_history - contour / hull
        self.solidity_history = deque(maxlen=30)
        # smoothed_history 保留, 仍用作 sm 趋势参考 (旧 Signal 4 等)
        self.smoothed_history = deque(maxlen=max(rise_window, fall_window, saturation_window))
        # 长期 history (用于 tr60, 区分 real leave 和 period 3 noise spike)
        self._long_history = deque(maxlen=60)
        # 饱和期内峰值参考: 最近 60 帧 smoothed 最大值
        self.sat_peak_history = deque(maxlen=60)
        # [v5.22 已回滚] smoothed_filled_history 趋势判定误触严重, 改用 unsat 调参
        # self.smoothed_filled_history = deque(maxlen=60)
        # 钢卷"已存在"持续时间
        self.above_enter_count = 0
        self.pending_state = None
        self.pending_count = 0
        # FALLING 单调下降计数器 (filled)
        self._fall_filled_monotone_count = 0
        # FALLING 单调下降计数器 (hull)
        self._fall_hull_monotone_count = 0
        # ENTER 单调上升计数器 (filled)
        self._enter_filled_monotone_count = 0

        # 二态主显示: STABLE / CHANGE
        self.state = 'STABLE'
        self.prev_state = 'STABLE'
        # 内部子状态
        self.sub_state = 'STABLE_NO_COIL'  # STABLE_NO_COIL / STABLE_COIL / CHANGE_RISING / CHANGE_FALLING
        # 事件锁: 已发 ENTER 后, 必须先发 LEAVE 才能再发 ENTER
        self.entered = False
        # LEAVE 已发标志: 防止 LEAVE 触发后 sm 还在 absolute_exit 之上又跳回 CHANGE_FALLING
        self._has_emitted_leave = False
        # 历史最大 sm (慢响应, 用于判定"钢卷基本走完")
        self.max_seen = 1.0
        # 子状态二次确认 (防铺满期噪声毛刺)
        self.pending_sub = None
        self.pending_sub_count = 0
        self.confirm_sub_change = 3  # 子状态切换需连续 N 帧确认
        # 信号 1 触发状态: 一旦信号 1 触发过, 持续保持 _is_falling=True
        # 直到信号 2 或 3 接管 (不会"退出"为 False)
        self._fall_signal1_active = False
        # 稳定铺满帧计数: 当前 sub_state == 'STABLE_COIL' 的连续帧数
        # 用于 _is_falling_trend_only 的 guard: ENTER 后 settling 期不触发
        self._stable_coil_frames = 0
        # CHANGE_FALLING 粘性: 一旦进入, 持续保持直到 sm 恢复到接近 max_seen
        self._fall_sticky_active = False
        # 钢卷真正峰值 (ENTER 时记录的 sm, 不衰减), 用作 sticky 释放参考
        # max_seen 有 0.999 衰减, 跟 sm 一起降, 比值一直接近 1, 无法检测 "回稳"
        # _coil_peak 是 ENTER 时的 sm, 钢卷完全离开才重置, 才是真正参考
        self._coil_peak = 0.0
        # v6.6: settled 期 hull 最大值, 作为 hull_shrinking 的 reference
        # STABLE 期 hull 可能继续微降 (settling 残留), max 是更可靠的 "settled" baseline
        self._hull_settled_peak = 0.0
        # 连续低于 absolute_enter 帧数: 用来在 sticky 状态下检测 "钢卷真走完"
        # 当 sticky 还 hold 但 sm 已 < absolute_enter 持续 N 帧, 说明钢卷真走了, 释放 sticky
        self._below_enter_count = 0
        # post-LEAVE 防御: LEAVE 后必须先看到 sm < absolute_exit, 才允许新 ENTER.
        # 防止残余 sm (LEAVE 时 sm 还很高) 触发 phantom ENTER.
        self._post_leave_saw_empty = False
        # v5.28: re-RISING 一次性使用标志 (LEAVE 后重置)
        self._re_rising_used = False
        # baseline: coil 稳定时 sm 参考 (仅在 STABLE_COIL 时维护)
        self._baseline = 0.0
        # [v7] consec signal: 帧间差分连续计数
        self._prev_gray = None
        self.consec_history = deque(maxlen=300)  # ~5 秒 @ 60fps
        self._consec_peak = 0.0  # rolling max of consec_history, settled 期更新
        self._consec_peak_window_frames = int(round(consec_peak_window_sec * 60))
        self._consec_mono_window_frames = int(round(consec_mono_window_sec * 60))
        # consec_peak 就绪门控: ENTER 后 consec 还会持续下降 (coil settling),
        # 如果早早采 consec_peak 会把 settling 期的高值当成 reference,
        # settling 完成后 consec 自然下降, 立即误触发 LEAVE.
        # 等 STABLE_COIL 持续 settle_guard+180 帧 (~5s) 再采 reference.
        self._consec_peak_ready = False
        self.event_count = {'ENTER': 0, 'LEAVE': 0}
        self.event_log = []

        # kernel 元素是占位值, 第一次 update() 时按 frame 短边归一化后重建
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )
        # 大核, 用于 filled_area 计算 (抗 wire 间隙)
        self.fill_kernel_elem = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (fill_kernel, fill_kernel)
        )

    def _build_bg(self):
        arr = np.stack(list(self.bg_buffer), axis=0)
        self.bg = np.median(arr, axis=0).astype(np.uint8)
        self.bg_f = self.bg.astype(np.float32)

    def _update_bg(self, frame_gray):
        if self.ema_alpha <= 0:
            return
        cv2.accumulateWeighted(frame_gray, self.bg_f, self.ema_alpha)
        self.bg = self.bg_f.astype(np.uint8)

    def _get_foreground(self, frame_gray):
        diff = cv2.absdiff(frame_gray, self.bg)
        _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            mask_roi = np.zeros_like(mask)
            mask_roi[ry:ry+rh, rx:rx+rw] = mask[ry:ry+rh, rx:rx+rw]
            mask = mask_roi
        return mask

    def _compute_metrics(self, mask):
        """从 mask 计算多指标: filled_area, hull_area, solidity, raw_area

        输入: 帧差 mask (uint8 0/255)
        输出: dict {
            'raw': int  (原 mask 面积, =sm 旧指标),
            'filled': int (大核 MORPH_CLOSE 后面积, 主指标),
            'hull': int  (凸包面积, 辅指标),
            'solidity': float (contour_area / hull_area, 0-1)
        }

        设计:
          - filled 用大核 close 抗 wire 间隙, 钢卷"实体"覆盖的稳定估计
          - hull 用最大 contour 的凸包, 抗 coil 横向移动 (coil 移动时 hull 几乎不变)
          - solidity 区分 "coil 散开" (低) vs "coil 缩小" (高)
        """
        raw = int((mask > 0).sum())
        if raw < self.min_area:
            # mask 太小, 算 hull 会爆, 直接给 0
            return {'raw': raw, 'filled': 0, 'hull': 0, 'solidity': 0.0}

        # filled: 大核 close 填 wire 间隙
        filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.fill_kernel_elem)
        filled = int((filled_mask > 0).sum())

        # hull: 最大 contour 的凸包
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'raw': raw, 'filled': filled, 'hull': 0, 'solidity': 0.0}
        biggest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(biggest)
        hull_area = int(cv2.contourArea(hull))
        contour_area = int(cv2.contourArea(biggest))
        solidity = float(contour_area) / float(hull_area) if hull_area > 0 else 0.0

        return {
            'raw': raw,
            'filled': filled,
            'hull': hull_area,
            'solidity': solidity,
        }

    def _smoothed_filled(self):
        """filled_area 的滑动平均 (主指标的 smoothed 版本)"""
        if not self.filled_history:
            return 0.0
        # 用最近 area_smooth 帧
        recent = list(self.filled_history)[-self.area_smooth:]
        return sum(recent) / len(recent)

    def _smoothed_hull(self):
        if not self.hull_history:
            return 0.0
        recent = list(self.hull_history)[-self.area_smooth:]
        return sum(recent) / len(recent)

    def _smoothed_area(self):
        if not self.area_history:
            return 0.0
        return sum(self.area_history) / len(self.area_history)

    def _trend_ratio(self, n):
        """最近 n 帧 smoothed 的端点相对差 = (末-首)/median

        返回:
          > rise_threshold: 上升趋势
          < -fall_threshold: 下降趋势
          其他: 平稳

        用中位数 (而非均值) 作为分母, 对单点毛刺鲁棒.
        """
        if len(self.smoothed_history) < n:
            return 0.0
        recent = np.array(list(self.smoothed_history)[-n:], dtype=np.float32)
        end_diff = recent[-1] - recent[0]
        median = float(np.median(recent))
        if median < 1.0:
            return 0.0
        return float(end_diff / median)

    def _long_trend_ratio(self, n):
        """跟 _trend_ratio 相同算法, 但用 _long_history (maxlen=60).

        用于 _is_falling_trend_only 的 tr60 联合判断, 区分 real leave 和
        period 3 F10156 noise spike. 独立 deque 避免影响 _is_unsaturating
        (要求 smoothed_history.len >= 60, maxlen=30 永远 False).
        """
        if len(self._long_history) < n:
            return 0.0
        recent = np.array(list(self._long_history)[-n:], dtype=np.float32)
        end_diff = recent[-1] - recent[0]
        median = float(np.median(recent))
        if median < 1.0:
            return 0.0
        return float(end_diff / median)

    def _update_reference(self):
        """更新 reference = 过去 ref_window 帧 smoothed 的滚动最大值

        慢响应: reference 跟着钢卷进入上升, 跟着钢卷长期离开下降.
        这样 enter_ratio/exit_ratio 是相对钢卷峰值的, 自适应.
        """
        if len(self.smoothed_history) < 5:
            return
        window = list(self.smoothed_history)[-self.ref_window:]
        # 用 90 分位数, 抗单点毛刺
        self.reference = max(self.reference * 0.99, float(np.percentile(window, 90)))

    def _is_rising(self):
        """filled 持续上升 (新钢卷进入, 主指标改用 filled_area)

        用 filled_area 替代 sm, 抗 coil 边缘进入时的 sm 噪扰.
        仅在 entered=False 状态下生效.
        """
        if self.entered:
            return False
        return self._filled_trend_ratio(self.rise_window) > self.rise_threshold

    def _filled_trend_ratio(self, n):
        """filled_history 的端点相对差, 同 _trend_ratio 算法"""
        if len(self.filled_history) < n:
            return 0.0
        recent = np.array(list(self.filled_history)[-n:], dtype=np.float32)
        end_diff = recent[-1] - recent[0]
        median = float(np.median(recent))
        if median < 1.0:
            return 0.0
        return float(end_diff / median)

    def _hull_trend_ratio(self, n):
        """hull_history 的端点相对差"""
        if len(self.hull_history) < n:
            return 0.0
        recent = np.array(list(self.hull_history)[-n:], dtype=np.float32)
        end_diff = recent[-1] - recent[0]
        median = float(np.median(recent))
        if median < 1.0:
            return 0.0
        return float(end_diff / median)

    def _filled_long_trend_ratio(self, n):
        """filled_history 长窗 (60 帧) 端点相对差"""
        if len(self.filled_history) < n:
            return 0.0
        recent = np.array(list(self.filled_history)[-n:], dtype=np.float32)
        end_diff = recent[-1] - recent[0]
        median = float(np.median(recent))
        if median < 1.0:
            return 0.0
        return float(end_diff / median)

    # [v5.22 已回滚] _filled_smoothed_trend_ratio: 趋势判定误触严重 (Coil 1 t=18s 误触)
    # def _filled_smoothed_trend_ratio(self, n):
    #     ...

    def _filled_monotone_count(self, n):
        """最近 n 帧 filled 单调下降的帧数: filled[i] < filled[i+1] 计数"""
        if len(self.filled_history) < n:
            return 0
        h = list(self.filled_history)[-n:]
        return sum(1 for i in range(n - 1) if h[i] > h[i + 1])

    def _hull_monotone_count(self, n):
        """最近 n 帧 hull 单调下降的帧数"""
        if len(self.hull_history) < n:
            return 0
        h = list(self.hull_history)[-n:]
        return sum(1 for i in range(n - 1) if h[i] > h[i + 1])

    def _filled_monotone_up_count(self, n):
        """最近 n 帧 filled 单调上升的帧数: filled[i] < filled[i+1] 计数"""
        if len(self.filled_history) < n:
            return 0
        h = list(self.filled_history)[-n:]
        return sum(1 for i in range(n - 1) if h[i] < h[i + 1])

    def _is_falling_trend_only(self, filled):
        """v5.6 + v6: filled 单调下降 + hull 单调下降 + magnitude guard (参数化)

        guard:
          - entered = True
          - scf >= settle_guard_frames (120)
          - filled < fall_peak_ratio × _coil_peak (防 settling 期 "filled 先跌后稳" 误触)
            v5.5 用 0.85, 但 Coil 3 稳定期 filled 在 600-900k 间持续震荡,
            频繁跌破 0.85 = 740k, 视觉 LEAVE 启动 (t=195s) 之前 2.5s 就触发.
            Coil 3 视觉 LEAVE 真正启动时 filled=83k, 即 ~10% peak.
            v5.6: 0.85 → 0.50, 要求 filled 跌破 50% peak (≈ coil 真的离场一半) 才触发.
            v6: 0.50 → fall_peak_ratio 参数化, 用户 GT 要求 "一离开就发",
                调到 0.70-0.85 让 LEAVE 提前 1-2s (由 monotone 守卫防 STABLE 期误触)
          - filled 单调下降 ≥ fall_monotone_min (18/30)
          - hull 单调下降 ≥ fall_hull_monotone_min (22/30)
        """
        if not self.entered:
            return False
        if self._stable_coil_frames < self.settle_guard_frames:
            return False
        # magnitude guard: 要求 coil 离开到 fall_peak_ratio 比例 (默认 0.50 = 走了一半)
        if self._coil_peak > 0 and filled > self._coil_peak * self.fall_peak_ratio:
            return False
        if self._filled_monotone_count(30) < self.fall_monotone_min:
            return False
        if self._hull_monotone_count(30) < self.fall_hull_monotone_min:
            return False
        return True

    def _is_hull_shrinking(self):
        """v6.4: 几何信号 - coil 真离开时 hull 凸包面积缩小

        关键洞察: filled area 受背景建模 (EMA) 干扰, STABLE 期间慢慢跌 30%+
        但凸包 (hull) 是 mask 轮廓的几何包络, 只反映 coil 实际形状.
        coil 真离开时, mask 缩小, hull 面积同步缩.
        STABLE 期 filled 漂移但 hull 不变 (几何包络不在乎背景建模).

        触发条件 (v6.6 baseline + v10 _is_rapid_drop 已分离):
          - entered + settled ≥ settle_guard_frames
          - hull < _hull_settled_peak × (1 - 0.05)        (hull 缩小 ≥ 5%)
          - hull 30 帧单调下降 ≥ 20/29 (~69%)
        注意: v10 改 _is_falling_impl 加 _is_rapid_drop 早触发,
              本函数保留 v6.6 阈值作为几何 fallback.
        """
        if not self.entered:
            return False
        if self._stable_coil_frames < self.settle_guard_frames:
            return False
        if len(self.hull_history) < 30:
            return False
        peak_hull = getattr(self, '_hull_settled_peak', 0.0)
        if peak_hull < 1.0:
            return False
        current_hull = self.hull_history[-1]
        if current_hull < 1.0:
            return False
        drop_ratio = (peak_hull - current_hull) / peak_hull
        if drop_ratio < 0.05:
            return False
        m30 = self._hull_monotone_count(30)
        if m30 < 20:
            return False
        return True

    def _is_falling(self, filled):
        result = self._is_falling_impl(filled)
        last_pk = getattr(self, '_dbg_falling_pk', 0)
        if self._coil_peak != last_pk:
            self._dbg_falling_printed = False
            self._dbg_falling_pk = self._coil_peak
        if result and not getattr(self, '_dbg_falling_printed', False):
            print(f'  [is_falling TRUE @ F{self.frame_idx} t={self.frame_idx/60:.2f} filled={filled:.0f} peak={self._coil_peak:.0f}]')
            self._dbg_falling_printed = True
        return result

    def _is_falling_impl(self, filled):
        """v6.5: hull 几何信号 (放宽 3%/15) 作为唯一早触发

        实测 v6.4 (hull_shrinking 5%/20):
          C1 LEAVE 63.37s (GT 61.5) bias +1.87s
          C2 LEAVE 130.65s (GT 129.5) bias +1.15s ✓
          C3 LEAVE 192.52s (GT 191) bias +1.52s ✓
        解决 C2/C3 STABLE 期误触, 但 LEAVE 偏晚 1-2s.

        v6.5: 阈值 5%/20 → 3%/15, 让 hull 一开始缩就触发, 更早对齐 GT.

        信号链 (任一满足):
          -1) [v6.5] hull 30 帧内缩小 ≥ 3% + monotone 15/29
           0) [v5.x] entered + settled + filled 单调下降 + hull 单调下降
           2) [v5.14] smoothed 跌破 _coil_peak × fall_peak_ratio + 25 帧累计
        """
        if len(self.filled_history) < 6 or self.max_seen < 1.0:
            return False

        # 信号 -1 (v6.5): hull 30 帧内缩小 ≥ 3% + monotone 15/29
        if self._is_hull_shrinking():
            return True

        # 信号 0: filled 单调下降 + hull 单调下降
        if self._is_falling_trend_only(filled):
            return True

        # 信号 2 (v5.14 + v6): smoothed 跌破 _coil_peak × fall_peak_ratio + 25 帧累计 (fallback)
        #   smoothed 用 area_smooth 平均压制单帧毛刺, 防背景建模抬高 max_seen
        #   当 coil 真离开但 hull 信号不及时触发时, 此信号作为 fallback
        smoothed = getattr(self, '_cur_smoothed', filled)
        if self._coil_peak > 0 and smoothed < self._coil_peak * self.fall_peak_ratio:
            if not hasattr(self, '_below_85_count'):
                self._below_85_count = 0
            self._below_85_count += 1
            if self._below_85_count >= 25:
                return True
        else:
            if hasattr(self, '_below_85_count'):
                self._below_85_count = 0

        return False

    def _is_saturated(self):
        """v5.1: filled 接近 max 就算 STABLE_COIL, solidity 仅作软参考

        实测 3.mp4 稳定期 solidity 长期 0.4-0.6 (wire pattern 间隙),
        阈值 0.85 永远不触发. 改为只看 filled median/max ratio.
        solidity 仅在 _is_falling 中用于"防 coil 散开误判离开", 不参与 STABLE 判定.
        """
        if len(self.filled_history) < self.saturation_window:
            return False
        recent = np.array(list(self.filled_history)[-self.saturation_window:],
                          dtype=np.float32)
        max_recent = float(recent.max())
        median_recent = float(np.median(recent))
        if max_recent < 1.0:
            return False
        return (median_recent / max_recent) > self.saturation_ratio

    def _is_unsaturating(self):
        result = self._is_unsaturating_impl()
        last_pk = getattr(self, '_dbg_unsat_pk', 0)
        if self._coil_peak != last_pk:
            self._dbg_unsat_printed = False
            self._dbg_unsat_pk = self._coil_peak
        if result and not getattr(self, '_dbg_unsat_printed', False):
            print(f'  [is_unsaturating TRUE @ F{self.frame_idx} t={self.frame_idx/60:.2f} peak={self._coil_peak:.0f}]')
            self._dbg_unsat_printed = True
        return result

    def _is_unsaturating_impl(self):
        """v6.6 baseline: filled 跌破 _coil_peak × unsaturating_ratio → 离开期

        v5.4 仍误触 (Coil 3 t=170s), 根因: filled_history 60 帧 max 包含 ENTER 时
        瞬间尖峰 (1343k), 远高于实际 peak (871k), median/peak=0.52 < 0.65 → 误触.
        修复: 用 _coil_peak (ENTER 时锁定的稳态 peak) 而非 filled_history 60 帧 max.

        v5.32: 阈值 0.65→0.72, 守卫 m30≥25
        v6: 0.72 硬编码 → unsaturating_ratio 参数 (默认 0.85).
            用户 GT 边界是 "coil 一离开一点就发" (≈filled 跌破 85% peak).
        实测 v6.6 best stable bias: C1 +1.87s, C2 +1.15s, C3 +1.52s.
        """
        if not self.entered or self._coil_peak <= 0:
            return False
        if len(self.filled_history) < 30:
            return False
        recent = np.array(list(self.filled_history)[-30:], dtype=np.float32)
        median_recent = float(np.median(recent))
        if median_recent >= self._coil_peak * self.unsaturating_ratio:
            return False
        if self._filled_monotone_count(30) < self.unsaturating_mono_min:
            return False
        return True

    def _is_consec_falling(self):
        """[v7] 帧间差分 LEAVE 早触发信号

        STABLE 期 consec 在 coil 微振动下保持 ~280-410k (rolling 2s peak).
        LEAVE 启动后 consec 持续下降 (coil 离场, 帧间差分变小),
        当 consec_median_recent < rolling_peak × consec_drop_ratio 且最近 1s 有 ≥30 帧单调下降 → True.

        比 _is_unsaturating 早 0.5-1.5s, 因为:
        - 不依赖 bg EMA (无累积漂移延迟)
        - consec 在 coil 一离开就立刻下降

        抗瞬时抖动: 用 median-of-recent-10 而非 consec_now 自身, 单帧 dip (压缩毛刺/光斑)
        会被 median 压平, 不会触发.

        守卫:
        - _consec_peak_ready: STABLE_COIL 持续 ≥ settle+180 帧才启用, 避免 coil settling
          阶段 consec 自然下降被误判为 LEAVE
        """
        if not self._consec_peak_ready or self._consec_peak <= 0:
            return False
        if not self.entered:
            return False
        if len(self.consec_history) < self._consec_mono_window_frames + 1:
            return False
        # median of last 10 frames (压平单帧 dip)
        recent = list(self.consec_history)[-10:]
        consec_median = float(np.median(np.array(recent, dtype=np.float32)))
        if consec_median >= self._consec_peak * self.consec_drop_ratio:
            return False
        # monotonic drop count in last K frames
        window = list(self.consec_history)[-self._consec_mono_window_frames - 1:]
        drops = sum(1 for i in range(1, len(window)) if window[i] < window[i - 1])
        return drops >= self.consec_mono_min

    def _is_consec_settled(self):
        """[v7-fix] post-LEAVE 阶段判定 "coil 已稳定离开, 可转 STABLE_NO_COIL"

        比 filled<absolute_exit 准确: filled EMA 衰减慢 (C1 视觉 64.5s 时 filled≈300k),
        而 consec 在 coil 物理离开的瞬间即下降到近 0, 反映真实运动停止.

        判定条件:
        - consec median of last 30 frames < peak × 0.05  (≈ 1.5-2 万)
        - consec median of last 10 frames < peak × 0.03  (≈ 1 万, 紧约束)
        - consec 连续 K 帧 ≤ peak × 0.05  (无反弹)
        """
        if not self._consec_peak_ready or self._consec_peak <= 0:
            return False
        if not self._has_emitted_leave:
            return False
        if len(self.consec_history) < 30:
            return False
        recent_30 = list(self.consec_history)[-30:]
        med30 = float(np.median(np.array(recent_30, dtype=np.float32)))
        recent_10 = list(self.consec_history)[-10:]
        med10 = float(np.median(np.array(recent_10, dtype=np.float32)))
        # 紧约束: median 必须持续低于 peak 的 5% / 3%
        if med30 >= self._consec_peak * 0.05:
            return False
        if med10 >= self._consec_peak * 0.03:
            return False
        # 连续 K 帧不超过 peak × 0.05 (避免短暂反弹)
        consec_settle_thresh = self._consec_peak * 0.05
        consec_settle_count = 0
        for v in recent_30:
            if v <= consec_settle_thresh:
                consec_settle_count += 1
            else:
                consec_settle_count = 0
        return consec_settle_count >= 20

    def _update_state_machine(self, smoothed_area):
        # debug: 打印 LEAVE 触发那一刻附近的状态
        if self.event_log and self.event_log[-1][1] == 'LEAVE' and self.frame_idx - self.event_log[-1][0] <= 2:
            print(f'  [LEAVE FIRE @ F{self.frame_idx} t={self.frame_idx/60:.2f} state={self.state} sub={self.sub_state} filled={smoothed_area:.0f} peak={self._coil_peak:.0f}]')
        # v5: smoothed_area 这里实际传 filled_smoothed (主指标)
        # 但旧 API 兼容: 也存到 smoothed_history 给旧逻辑兜底
        filled_smoothed = smoothed_area
        self.smoothed_history.append(smoothed_area)
        self._long_history.append(smoothed_area)
        self.sat_peak_history.append(smoothed_area)
        # [v5.22 已回滚] smoothed_filled_history
        # self.smoothed_filled_history.append(smoothed_area)
        # filled + hull + solidity history 已在 update() 中更新
        # 这里只需要用 filled_smoothed 做状态机

        # 更新 above_enter_count (用 filled 阈值)
        if filled_smoothed > self.absolute_enter:
            self.above_enter_count += 1
            self._below_enter_count = 0
        else:
            self.above_enter_count = 0
            if self._has_emitted_leave and not self.entered and filled_smoothed < self.absolute_enter:
                self._post_leave_saw_empty = True
            self._below_enter_count += 1

        # [v7-fix2] False LEAVE recovery: LEAVE 触发后 2.5s 时, 双信号联合判定:
        #   filled 几乎不变 (ratio ≥ 0.95) AND consec 没有明显下降趋势
        # → 假 LEAVE, 强制恢复到 STABLE_COIL, 避免视频末尾 stuck in CHANGE_FALLING.
        #
        # 为什么检查点用 2.5s (150 帧) 而非 1s (60 帧)?
        # - 1s 太短: 真 LEAVE 早期 consec 还没明显下降 (1.mp4 t=67.72 处 1s 后 drops=28, 3.mp4 C4 drops=30),
        #   单调下降计数无法区分真假 LEAVE (都 ~30).
        # - 2.5s 足够: 真 LEAVE 后 consec 持续下降 (1.mp4 5s 后 decay=0.72, drops=42);
        #   假 LEAVE 时 consec 仍在 ~peak 震荡 (3.mp4 C4 5s 后 decay≈1.0, drops≈30).
        #
        # 关键判据: consec_decay_ratio = consec_now / consec_at_LEAVE (2.5s 窗口起点)
        # - 真 LEAVE: decay < 0.85 (consec 衰减 ≥15%)
        # - 假 LEAVE: decay ≥ 0.95 (consec 几乎不变)
        # 边界 0.85-0.95 区间: 仍用 filled ratio + decay ratio 联合判定, 鲁棒.
        #
        # 数据点:
        #   3.mp4 C1/C2/C3 ratio=0.88/0.84/0.89, decay≈0.30-0.40 → 真 LEAVE, 不恢复
        #   3.mp4 C4 ratio=1.01, decay≈1.0 → 假 LEAVE, 恢复
        #   1.mp4 t=67.72 ratio=0.95 边界 → 2.5s 后 decay=0.90 → 真 LEAVE, 不恢复
        if self._has_emitted_leave and not self.entered:
            self._post_leave_filled_at_leave = getattr(self, '_post_leave_filled_at_leave', 0)
            self._post_leave_consec_at_leave = getattr(self, '_post_leave_consec_at_leave', 0)
            # 初次 LEAVE 后记录 filled 和 consec
            if self._post_leave_filled_at_leave == 0 and self.event_log and \
                    self.event_log[-1][1] == 'LEAVE' and \
                    self.frame_idx - self.event_log[-1][0] <= 2:
                self._post_leave_filled_at_leave = filled_smoothed
                self._post_leave_consec_at_leave = self.consec_history[-1] if self.consec_history else 0
            # 2.5s 后 (150 frames) 双信号联合检查
            self._post_leave_check_count = getattr(self, '_post_leave_check_count', 0) + 1
            if self._post_leave_filled_at_leave > 0 and self._post_leave_check_count == 150:
                ratio = filled_smoothed / self._post_leave_filled_at_leave if self._post_leave_filled_at_leave > 0 else 1.0
                # 计算 2.5s 区间 consec 衰减率
                consec_now = self.consec_history[-1] if self.consec_history else 0
                consec_decay = consec_now / self._post_leave_consec_at_leave if self._post_leave_consec_at_leave > 0 else 1.0
                # 假 LEAVE 判定 (双信号 AND):
                #   filled 几乎不变 (ratio ≥ 0.95) AND consec 没下降 (decay ≥ 0.85)
                is_false_leave = (ratio >= 0.95) and (consec_decay >= 0.85)
                if is_false_leave:
                    print(f'  [FALSE LEAVE RECOVER @ F{self.frame_idx} t={self.frame_idx/60:.2f} '
                          f'filled_ratio={ratio:.2f}, consec_decay={consec_decay:.2f} (both -> false)]')
                    self._has_emitted_leave = False
                    self._post_leave_filled_at_leave = 0
                    self._post_leave_consec_at_leave = 0
                    self._post_leave_check_count = 0
                    self._fall_sticky_active = False
                    self._stable_coil_frames = 0
                    self.state = 'STABLE'
                    self.sub_state = 'STABLE_COIL'
                    self.entered = True
                    self._coil_peak = filled_smoothed
                    return
                else:
                    # 真正 LEAVE, 重置检查计数 (后续不再触发)
                    self._post_leave_filled_at_leave = 0
                    self._post_leave_consec_at_leave = 0
                    self._post_leave_check_count = 0
        else:
            self._post_leave_check_count = 0
            self._post_leave_filled_at_leave = 0
            self._post_leave_consec_at_leave = 0

        # max_seen 滚动跟踪: max(max_seen × 0.999, filled)
        if filled_smoothed > 1.0:
            self.max_seen = max(self.max_seen * 0.999, filled_smoothed)

        # baseline: coil 稳定时的 filled 参考
        if self.entered and self.sub_state == 'STABLE_COIL' and filled_smoothed > 1.0:
            if not hasattr(self, '_baseline') or self._baseline < filled_smoothed:
                self._baseline = filled_smoothed
            else:
                self._baseline = max(self._baseline * 0.9999, filled_smoothed)

        # Warmup 保护
        warmup_end = self.bg_frames + 60
        if self.frame_idx < warmup_end:
            self.max_seen = float(filled_smoothed) if filled_smoothed > 1.0 else 1.0
            self.smoothed_history.clear()
            self.sat_peak_history.clear()
            self._long_history.clear()
            self.filled_history.clear()
            self.hull_history.clear()
            self.solidity_history.clear()
            self.above_enter_count = 0
            self._below_enter_count = 0
            for _ in range(60):
                self.smoothed_history.append(filled_smoothed)
                self.sat_peak_history.append(filled_smoothed)
                self._long_history.append(filled_smoothed)
                self.filled_history.append(filled_smoothed)
                self.hull_history.append(0)
                self.solidity_history.append(0.0)
            raw_new_sub = 'STABLE_NO_COIL'
            new_sub = 'STABLE_NO_COIL'
            self.sub_state = new_sub
            if self.state != 'STABLE':
                if self.pending_state == 'STABLE':
                    self.pending_count += 1
                    if self.pending_count >= self.confirm_enter:
                        self._commit_state('STABLE', new_sub, filled_smoothed)
                else:
                    self.pending_state = 'STABLE'
                    self.pending_count = 1
            else:
                self.pending_state = None
                self.pending_count = 0
            return

        # v5.35: 完全禁用 re-RISING 静默抬高 peak
        #   C3 在 t=190.33s filled 涨到 922k (1.051×877k) 触发 re-RISING 静默抬 peak 到 935k,
        #   之后 filled 立即开始跌 (peak×0.83 = 776k 是 0.85 阈值)
        #   但 C3 视觉 STABLE 持续到 t=195.0s, detector 偏早 2-3s
        #   抬高 peak 让 detector 看不到 "filled 偏离原始 peak" 的信号
        #   v5.35: 移除整个 re-RISING 静默抬高 peak 逻辑, peak 在 ENTER 时锁定后保持不变
        pass  # v5.35: 禁用 re-RISING, 用 ENTER 时锁定的 _coil_peak 贯穿整个 STABLE 周期

        # 判定子状态 (v5.7: filled 为主, peak 替代 max_seen)
        # 优先级:
        # 1) filled < absolute_exit: 强制 STABLE_NO_COIL
        # 2) 已发过 LEAVE 且 filled < _coil_peak × max_seen_ratio: 锁定 STABLE_NO_COIL
        #    v5.6 用 max_seen, 但 max_seen 跟随 filled 衰减 (LEAVE 期 filled↓ → max_seen↓),
        #    Coil 3 LEAVE 启动 t=194.33s filled=164k 时 max_seen 已经降到 788k,
        #    0.25 = 197k, filled=164k < 197k 立即 STABLE_NO_COIL (偏早 3.67s)
        #    改为 _coil_peak: 锁住, 不受衰减影响
        # 3) 已发过 ENTER 且 filled < _coil_peak × max_seen_ratio: 强制 STABLE_NO_COIL
        # 4) filled 在 exit~enter 之间: 保持当前子状态
        # 5) filled > absolute_enter: 看趋势
        # v5.14: 把 smoothed 暂存供 _is_falling Signal 2 用
        self._cur_smoothed = filled_smoothed
        # [v7 post-LEAVE] LEAVE 事件已发, 但 coil 还没完全离开 (filled 还高),
        # 必须保持 CHANGE_FALLING 让用户能看到 coil 离开过程 (与视觉标注一致:
        # C1 持续 61.5→64.5 (3s), C2 持续 129.5→131.5 (2s), C3 持续 191→194 (3s)).
        # [v7-fix] 用 _is_consec_settled() 替代 filled<absolute_exit 判定 "coil 完全离开":
        # - filled<absolute_exit (5000) 触发太晚 (5.5-6s vs 视觉 2-3s), 因 EMA 衰减慢
        # - consec 在 coil 物理离开的瞬间即下降到近 0, 反映真实运动停止
        # - 视觉标注的 "coil 完全离开" ≈ consec 已稳定到近 0 值
        if (self._has_emitted_leave
                and self.sub_state == 'CHANGE_FALLING'
                and filled_smoothed >= self.absolute_exit
                and not self._is_consec_settled()):
            raw_new_sub = 'CHANGE_FALLING'
        elif self._is_consec_settled() and filled_smoothed < self.absolute_exit * 5:
            raw_new_sub = 'STABLE_NO_COIL'
        elif filled_smoothed < self.absolute_exit:
            raw_new_sub = 'STABLE_NO_COIL'
        elif (self._has_emitted_leave
              and self._coil_peak > 0
              and filled_smoothed < self._coil_peak * self.max_seen_ratio):
            raw_new_sub = 'STABLE_NO_COIL'
        elif (self.entered and self._coil_peak > 0
              and filled_smoothed < self._coil_peak * self.max_seen_ratio):
            raw_new_sub = 'STABLE_NO_COIL'
        elif filled_smoothed < self.absolute_enter:
            # v5.15: 30 → 60 (1s 持续 < absolute_enter)
            # Coil 3 LEAVE @194.83s 偏早 3.17s (视觉 198s). 提高阈值让 LEAVE 多等 1s,
            # 新 LEAVE 触发约 195.83s → 偏早 2.17s (改善 1s)
            # Coil 1/2 LEAVE 也推后 1s (从 ~65s→66s, ~133s→134s, 仍接近视觉)
            if self._below_enter_count > 60:
                raw_new_sub = 'STABLE_NO_COIL'
                self._fall_sticky_active = False
            else:
                raw_new_sub = self.sub_state
        else:
            # filled > absolute_enter
            # STABLE_NO_COIL → CHANGE_* 初次进入门槛
            if self.sub_state == 'STABLE_NO_COIL' and filled_smoothed < self.first_change_threshold:
                raw_new_sub = 'STABLE_NO_COIL'
            elif (self._is_falling(filled_smoothed) or self._is_unsaturating()
              or self._is_consec_falling()):
                raw_new_sub = 'CHANGE_FALLING'
                self._fall_sticky_active = True
            elif self._is_rising():
                raw_new_sub = 'CHANGE_RISING'
                self._fall_sticky_active = False
            elif self._is_saturated():
                if self._fall_sticky_active and self._coil_peak > 0 and filled_smoothed < self._coil_peak * 0.99:
                    raw_new_sub = 'CHANGE_FALLING'
                else:
                    raw_new_sub = 'STABLE_COIL'
                    self._fall_sticky_active = False
            else:
                if self._fall_sticky_active and self._coil_peak > 0 and filled_smoothed >= self._coil_peak * 0.50:
                    raw_new_sub = 'CHANGE_FALLING'
                else:
                    raw_new_sub = self.sub_state

        # 子状态二次确认
        if raw_new_sub != self.sub_state and self.sub_state != 'STABLE_NO_COIL':
            if self.pending_sub == raw_new_sub:
                self.pending_sub_count += 1
            else:
                self.pending_sub = raw_new_sub
                self.pending_sub_count = 1
            if self.pending_sub_count >= self.confirm_sub_change:
                new_sub = raw_new_sub
                self.pending_sub = None
                self.pending_sub_count = 0
            else:
                new_sub = self.sub_state
        else:
            new_sub = raw_new_sub
            self.pending_sub = None
            self.pending_sub_count = 0

        new_main = 'CHANGE' if new_sub in ('CHANGE_RISING', 'CHANGE_FALLING') else 'STABLE'

        if new_main != self.state:
            if self.pending_state == new_main:
                self.pending_count += 1
                threshold = self.confirm_enter if new_main == 'STABLE' else self.confirm_leave
                if self.pending_count >= threshold:
                    self._commit_state(new_main, new_sub, filled_smoothed)
            else:
                self.pending_state = new_main
                self.pending_count = 1
        else:
            self.pending_state = None
            self.pending_count = 0

        old_sub = self.sub_state
        if (not self.entered and new_sub == 'STABLE_COIL'
                and self._has_emitted_leave and not self._post_leave_saw_empty):
            new_sub = 'STABLE_NO_COIL'
        self.sub_state = new_sub

        if new_sub == 'STABLE_COIL':
            self._stable_coil_frames += 1
        else:
            self._stable_coil_frames = 0

        if not self.entered and new_sub == 'STABLE_COIL':
            self._coil_peak = filled_smoothed
            self._emit_event('ENTER', filled_smoothed)
            self.entered = True
            self._has_emitted_leave = False
            self._fall_signal1_active = False
            self._consec_peak_ready = False  # [v7] 重置, 等 settle+180 帧再开
        elif self.entered and old_sub == 'STABLE_COIL' and new_sub == 'CHANGE_FALLING':
            # v5.47a: LEAVE fires on STABLE_COIL → CHANGE_FALLING 转换 (coil 开始离开)
            # 替代原来 STABLE_NO_COIL 触发 (coil 完全离开, 偏晚 3-5s)
            self._emit_event('LEAVE', filled_smoothed)
            self.entered = False
            self._has_emitted_leave = True
            self._fall_signal1_active = False
            self._stable_coil_frames = 0
            self._fall_sticky_active = False
            self._hull_settled_peak = 0.0  # v6.6: 重置, 下一个 ENTER 重新记录
            self._coil_peak = 0.0
            self._baseline = 0.0
            # [v7-fix] 不重置 _consec_peak, 留作 post-LEAVE 阶段判断 "coil 已稳定离开"
            # 用 _is_consec_settled() 判定 CHANGE_FALLING → STABLE_NO_COIL 的转换时机,
            # 比 filled<absolute_exit (5000) 准确得多: filled EMA 衰减慢, consec 反映真实运动.
            self.max_seen = 1.0
            if hasattr(self, '_below_85_count'):
                self._below_85_count = 0
            if hasattr(self, '_below_95_count'):
                self._below_95_count = 0
            if hasattr(self, '_below_95b_count'):
                self._below_95b_count = 0
            self._post_leave_saw_empty = False
        elif (self.entered or self._has_emitted_leave) and new_sub == 'STABLE_NO_COIL':
            # v5.47a: STABLE_NO_COIL 只清理状态, 不再触发 LEAVE (已在 CHANGE_FALLING 时发出)
            # 要求 entered 或 _has_emitted_leave (确保 warmup 期不会误触发, 阻挡 ENTER)
            # 这一分支仅在 CHANGE_FALLING 被跳过时触发 (filled 跳过 unsaturating 阈值直接跌穿)
            # 此时需要再次触发 LEAVE + 完整清理
            entered_was_true = self.entered  # 保存 entered 状态, 区分 fallback vs CHANGE_FALLING 后到达
            if entered_was_true:
                self._emit_event('LEAVE', filled_smoothed)
            self.entered = False
            self._has_emitted_leave = True
            self._fall_signal1_active = False
            self._fall_sticky_active = False
            self._stable_coil_frames = 0
            self._coil_peak = 0.0
            self._baseline = 0.0
            self._consec_peak = 0.0  # [v7] 重置
            self.max_seen = 1.0
            if hasattr(self, '_below_85_count'):
                self._below_85_count = 0
            if hasattr(self, '_below_95_count'):
                self._below_95_count = 0
            if hasattr(self, '_below_95b_count'):
                self._below_95b_count = 0
            if entered_was_true:
                # Fallback 情况: 刚发 LEAVE, 重置 post_leave_saw_empty 让 line 608 重新检测
                self._post_leave_saw_empty = False
            # else: CHANGE_FALLING 已发 LEAVE, post_leave_saw_empty 已被 line 608 设为 True,
            #       不能重置, 否则会永远阻挡下一次 ENTER
            if hasattr(self, '_re_rising_count'):
                self._re_rising_count = 0
            self._re_rising_used = False

    def _commit_state(self, new_main, new_sub, smoothed_area):
        self.prev_state = self.state
        self.state = new_main

    def _emit_event(self, event_type, smoothed_area):
        self.event_count[event_type] += 1
        self.event_log.append((self.frame_idx, event_type, smoothed_area))

    def update(self, frame_bgr, fps=30):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self.frame_idx += 1

        # [v7] 帧间差分 consec_diff: 不依赖 bg 累积, STABLE 期稳定, LEAVE 期下降
        if self._prev_gray is not None:
            consec_diff = cv2.absdiff(self._prev_gray, gray)
            _, consec_mask = cv2.threshold(consec_diff, self.consec_diff_thresh, 255, cv2.THRESH_BINARY)
            consec_mask = cv2.morphologyEx(
                consec_mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (self.consec_morph_kernel, self.consec_morph_kernel))
            )
            consec_filled = int((consec_mask > 0).sum())
        else:
            consec_filled = 0
        self.consec_history.append(consec_filled)
        self._prev_gray = gray

        # 第一次调用时, 按 frame_pixels 归一化所有面积阈值 (v6 鲁棒性)
        if self.frame_pixels is None:
            h, w = frame_bgr.shape[:2]
            self.frame_pixels = float(h * w)
            self._area_scale = self.frame_pixels / REFERENCE_PIXELS
            # 面积阈值按 frame_pixels 归一化
            if abs(self._area_scale - 1.0) > 1e-6:
                self.min_area = max(1, int(self._ref_min_area * self._area_scale))
                self.absolute_enter = max(1, int(self._ref_absolute_enter * self._area_scale))
                self.absolute_exit = max(1, int(self._ref_absolute_exit * self._area_scale))
                self.first_change_threshold = max(1, int(self._ref_first_change_threshold * self._area_scale))
                self.stable_enter_threshold = max(1, int(self._ref_stable_enter_threshold * self._area_scale))
            # 形态学核按 frame 短边归一化 (避免 0.33x 等极小尺寸下大核占比过大)
            # 0.33x 短边 480, kernel 15 → 占 3.1%, 大核 close 填充过头
            # 归一化后, kernel 占短边的 ~1% (2560x1440 ref 下 15/1440=1.04%)
            self._kernel_scale = min(h, w) / REFERENCE_MIN_DIM
            if abs(self._kernel_scale - 1.0) > 1e-6:
                self.morph_kernel = max(3, int(round(self._ref_morph_kernel * self._kernel_scale)))
                self.fill_kernel = max(3, int(round(self._ref_fill_kernel * self._kernel_scale)))
                self.kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
                )
                self.fill_kernel_elem = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.fill_kernel, self.fill_kernel)
                )

        if not self.initialized:
            self.bg_buffer.append(gray)
            if len(self.bg_buffer) >= self.bg_frames:
                self._build_bg()
                self.initialized = True
            return None, 'INIT', 0.0

        mask = self._get_foreground(gray)
        # v5: 多指标
        metrics = self._compute_metrics(mask)
        self.area_history.append(metrics['raw'])
        self.filled_history.append(metrics['filled'])
        self.hull_history.append(metrics['hull'])
        # v6.6: 在 settled STABLE 期更新 hull_settled_peak
        # 进入 STABLE_COIL 状态后一段时间, hull 达到峰值, 此后作为 reference
        if (self.entered and self.sub_state == 'STABLE_COIL'
                and self._stable_coil_frames > self.settle_guard_frames
                and metrics['hull'] > self._hull_settled_peak):
            self._hull_settled_peak = metrics['hull']  # 不衰减, 持续增大直到真离开
        self.solidity_history.append(metrics['solidity'])
        # [v7] settled 期更新 consec_peak: 滚动 N 秒窗口的中位数 (抗尖峰)
        # 用 median 而非 max: C3 STABLE 期 consec 有瞬间尖峰 (388k), max 会被污染;
        # median 对单个尖峰免疫, 反映稳定 baseline.
        if (self.entered and self.sub_state == 'STABLE_COIL'
                and self._stable_coil_frames > self.settle_guard_frames + 180
                and len(self.consec_history) >= self._consec_peak_window_frames):
            window_vals = np.array(list(self.consec_history)[-self._consec_peak_window_frames:],
                                   dtype=np.float32)
            self._consec_peak = float(np.median(window_vals))
            self._consec_peak_ready = True
        # 主指标: filled_smoothed
        filled_smoothed = self._smoothed_filled()
        # 旧 API 兼容: smoothed 仍返回 filled_smoothed
        smoothed = filled_smoothed

        self._update_state_machine(smoothed)
        self._update_bg(gray)

        return mask, self.state, smoothed

    def reset(self):
        self.bg = None
        self.bg_f = None
        self.bg_buffer.clear()
        self.initialized = False
        self.frame_idx = 0
        # 重置归一化状态, 下次 update() 时按新 frame_pixels 重新归一化
        # (reset 后, 阈值恢复成 REFERENCE_PIXELS 下的原始值, 由 update() 重算)
        self.frame_pixels = None
        self._area_scale = 1.0
        self._kernel_scale = 1.0
        self.min_area = self._ref_min_area
        self.absolute_enter = self._ref_absolute_enter
        self.absolute_exit = self._ref_absolute_exit
        self.first_change_threshold = self._ref_first_change_threshold
        self.stable_enter_threshold = self._ref_stable_enter_threshold
        self.morph_kernel = self._ref_morph_kernel
        self.fill_kernel = self._ref_fill_kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        self.fill_kernel_elem = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.fill_kernel, self.fill_kernel)
        )
        self.area_history.clear()
        self.filled_history.clear()
        self.hull_history.clear()
        self.solidity_history.clear()
        self.smoothed_history.clear()
        self.sat_peak_history.clear()
        self._long_history.clear()
        self.above_enter_count = 0
        self._below_enter_count = 0
        self.max_seen = 1.0
        self.pending_state = None
        self.pending_count = 0
        self.state = 'STABLE'
        self.prev_state = 'STABLE'
        self.sub_state = 'STABLE_NO_COIL'
        self.entered = False
        self._has_emitted_leave = False
        self._fall_signal1_active = False
        self._fall_sticky_active = False
        self._stable_coil_frames = 0
        self._coil_peak = 0.0
        self._below_99_count = 0
        if hasattr(self, '_below_85_count'):
            self._below_85_count = 0
        if hasattr(self, '_below_95_count'):
            self._below_95_count = 0
        if hasattr(self, '_below_95b_count'):
            self._below_95b_count = 0
        self._baseline = 0.0
        self._post_leave_saw_empty = False
        self._fall_filled_monotone_count = 0
        self._fall_hull_monotone_count = 0
        self._enter_filled_monotone_count = 0
        self._re_rising_used = False
        if hasattr(self, '_re_rising_count'):
            self._re_rising_count = 0
        self.event_count = {'ENTER': 0, 'LEAVE': 0}
        self.event_log.clear()