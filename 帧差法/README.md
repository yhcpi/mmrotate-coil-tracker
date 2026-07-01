# 帧差法钢卷检测

## 场景
- **相机**: 工业相机, 固定俯拍, 静止
- **钢卷**: 传送带从下方传过, 运动
- **目标**: 检测钢卷进出视野 (ENTER / LEAVE 事件)

## 方案: 帧差法 + 端点相对差状态机 (二态 STABLE / CHANGE + 4 子状态)

### 思路
1. **背景建模**: 用前 N 帧中位数建一个"无钢卷背景"
2. **帧差**: 当前帧 - 背景 → 前景 mask (钢卷区域)
3. **形态学**: 开闭运算去噪声、填空洞
4. **面积平滑**: mask 像素数做 15 帧滑动平均 → smoothed
5. **主状态机** (二态 STABLE / CHANGE):
   - STABLE: 画面稳定 (无钢卷 或 钢卷铺满稳定)
   - CHANGE: 钢卷正在进入或离开 (有趋势)
6. **子状态** (内部用, 用于发事件):
   - STABLE_NO_COIL: 无钢卷
   - STABLE_COIL: 钢卷铺满稳定
   - CHANGE_RISING: 钢卷正在进入
   - CHANGE_FALLING: 钢卷正在离开

### 状态语义

| 状态 | 含义 | 显示 |
|---|---|---|
| STABLE | 画面稳定 (无钢卷 或 钢卷铺满稳定) | 绿 |
| CHANGE | 钢卷正在进入或离开 (mask 面积在变) | 橙 |

### 子状态判定

- smoothed < absolute_exit (500) → STABLE_NO_COIL
- smoothed < max_seen × 0.50 → STABLE_NO_COIL (钢卷基本走完)
- smoothed > absolute_enter (3000) 且 _is_falling 或 _is_unsaturating → CHANGE_FALLING
- smoothed > absolute_enter 且 _is_rising → CHANGE_RISING
- smoothed > absolute_enter 且 _is_saturated (median/max > 0.8) → STABLE_COIL

### 关键机制

- `_is_falling` 三信号联合 (任一触发):
  1. signal 1: entered 状态下 5 帧内 sm 涨 2% (捕捉钢卷开始移动的边缘效应)
  2. signal 2: sm < max_seen × 0.85 (15% 跌幅, 抗背景慢速漂移)
  3. signal 3: sm < max_seen × 0.95 + tr15 < -0.04 (双指标联合)
- `_is_unsaturating`: 60 帧 median < max × 0.80 → 钢卷开始离开 (抗短反弹, 抗背景慢速漂移)
- `_is_saturated`: median/max > 0.8 → 钢卷已经铺满
- `_is_rising` / `_is_falling`: 端点相对差阈值 (短窗口, 快速响应)
- `entered` 锁: 第一次进入 STABLE_COIL 发 ENTER 后锁住, 回到 STABLE_NO_COIL 才解锁
- `confirm_enter=3, confirm_leave=3`: 非对称二次确认 (短, 快速响应)
- `_fall_signal1_active` 锁: signal 1 触发后保持, 防 5 帧 d5 短暂回落退出; ENTER/LEAVE 时重置

### 事件

- **ENTER**: 第一次进入 STABLE_COIL (钢卷来了)
- **LEAVE**: 从 STABLE_COIL 回到 STABLE_NO_COIL (钢卷走了)

### 文件
- `frame_diff_detector.py` - 核心检测器
- `framediff_runner.py` - 批量跑视频, dump 事件
- `demo.py` - 可视化 demo
- `synthetic_test.py` - 合成测试视频
- `tune.py` - 自动调参
- `gt_runner.py` - 老方法 ground truth (OBB + RailCoilDetector)
- `compare.py` - 事件匹配对比

## 用法

```bash
# 跑视频 (默认参数)
python framediff_runner.py <视频>

# 可视化 demo (启动后按 d 画 ROI, 拖鼠标画矩形后 enter 确认)
python demo.py <视频> --output out.mp4

# 全屏模式 (无 ROI)
python demo.py <视频> --no-show --output out.mp4

# 跑合成测试
python synthetic_test.py --quick

# 对比老方法 (前提: 已经跑过 gt_runner.py)
python compare.py <视频路径去扩展名>

# 调参
python tune.py <视频>
```

### demo.py 按键

- `q` 退出
- `r` 重置检测器
- `p` 暂停/继续
- `s` 保存当前帧为 PNG
- `d` 进入 ROI 画框模式 (拖鼠标画矩形, enter 确认, esc 取消, c 全屏)
  - 画的 ROI 自动保存到 `<视频>.roi.txt`, 下次启动自动加载

## 能力边界 (重要!)

帧差法本质是"像素变化"检测, 它能识别的是 mask 面积的上升沿和下降沿.

### 适用场景
- 钢卷之间有**明显的空场** (mask 面积回到 0)
- 单个钢卷存在时间任意长, 但持续期间 mask 面积不变

### 不适用场景
- 钢卷**无缝衔接** (前一卷还没完全离开, 下一卷已进入) → mask 面积始终高 → 帧差法识别为 1 个连续事件
- 钢卷**完全静止**停在视野中央 → 没有帧差 → 不触发

### 与老方法的对比

老方法 (OBB + RailCoilDetector) 通过**钢轨数量**判定钢卷存在, 即使钢卷无缝衔接也能区分. 帧差法只能看 mask 面积, 对相邻钢卷无能为力.

**5.mp4 对比** (99 秒, 1 个钢卷持续存在):
- 老方法: 1 个 ENTER @F375 (简化后) + 多个内部状态切换
- 帧差法: 1 个 ENTER @F375 ✓

**2.mp4 对比** (20 秒, 2 个钢卷紧密相邻):
- 老方法: 2 个 ENTER @F49 / F468 (区分了)
- 帧差法: 1 个 ENTER @F51 (漏检第 2 个, 因为 mask 没断)

## 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `bg_frames` | 30 | 背景建模帧数 |
| `diff_thresh` | 20 | 帧差灰度阈值 |
| `min_area` | 1500 | 最小前景面积 |
| `ema_alpha` | 0.01 | 背景 EMA 更新速度 |
| `roi` | None | 限制检测范围 (x, y, w, h) |
| `area_smooth` | 15 | mask 面积滑动平均窗口 |
| `absolute_enter` | 3000 | smoothed > 此值视为"有钢卷" |
| `absolute_exit` | 500 | smoothed < 此值视为"无钢卷" (发 LEAVE) |
| `rise_window` | 30 | 上升趋势窗口 (帧) |
| `fall_window` | 20 | 下降趋势窗口 (帧) |
| `rise_threshold` | 0.08 | 上升端点相对差阈值 |
| `fall_threshold` | 0.05 | 下降端点相对差阈值 |
| `rising_window` | 60 | 钢卷"刚出现"窗口 (此期间 _is_rising 才是真 ENTER) |
| `confirm_enter` | 3 | CHANGE→STABLE 需连续确认帧数 (非对称: 离开期早响应) |
| `confirm_leave` | 3 | STABLE→CHANGE 需连续确认帧数 |
| `confirm_sub_change` | 3 | 子状态切换需连续确认帧数 (抗铺满期噪声) |
| `max_seen_ratio` | 0.50 | sm < max_seen × 此值 → STABLE_NO_COIL (钢卷基本走完, 发 LEAVE) |
| `max_decay` | 0.999 | max_seen 每帧衰减系数 (慢响应) |
| `saturation_window` | 30 | 算 saturation ratio 的窗口 |
| `saturation_ratio` | 0.80 | median/max > 此值 视为已铺满 (进入完成) |
| `unsaturating_ratio` | 0.80 | 60 帧 median < max × 此值 视为开始离开 (抗背景慢速漂移) |

## 验证

```
synthetic 3 钢卷 (5秒间隔):      ENTER=3, LEAVE=3 ✓
synthetic 3 钢卷 (严格参数):     ENTER=3, LEAVE=3 ✓
synthetic 3 钢卷 (无 ROI):       ENTER=3, LEAVE=3 ✓
synthetic 4 钢卷 (7.5秒压力):    ENTER=4, LEAVE=4 ✓
5.mp4 (99秒, 1 个钢卷, ROI):    ENTER=1 @F585, LEAVE=1 @F2886, STABLE @F2888 ✓
2.mp4 (20秒, 2 紧邻钢卷):       ENTER=1 (漏检第 2 个, 已知限制) ⚠

5.mp4 ROI 时间线 (默认参数):
  F1-F30:      INIT (30 帧)
  F31-F378:    STABLE (背景)
  F379-F586:   CHANGE (进入期, 208 帧)
  F587-F2685:  STABLE (铺满期, 2099 帧, 0 抖动)
  F2686-F2887: CHANGE (离开期, 202 帧)
  F2888-F2983: STABLE (走完, 1min36s)
```