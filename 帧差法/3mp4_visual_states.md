# 3.mp4 视觉状态标注 (视觉能力完整重判版)

| 状态 | 含义 | 视觉特征 |
|---|---|---|
| `STABLE_NO_COIL` | 转轴空 | 仅见辊面 + 顶部细小残余 |
| `CHANGE_RISING` | 钢卷进入期 | 从底部/侧边进入, 持续填充 |
| `STABLE_COIL` | 钢卷铺满 | 体积稳定 (形状可能微动, 但 sm ≈ pk) |
| `CHANGE_FALLING` | 钢卷离开期 | 钢卷明显缩小, 部分区域清空 |

## 关键时间点 (v4 完整视觉重判)

| 事件 | 视觉边界 |
|---|---|
| **Coil 1** | |
| CHANGE_RISING 开始 | t=10.5s |
| STABLE_COIL 开始 | t=13.5s |
| CHANGE_FALLING 开始 | **t=61.5s** |
| STABLE_NO_COIL 开始 | t=64.5s |
| **Coil 2** | |
| CHANGE_RISING 开始 | t=79.5s |
| STABLE_COIL 开始 | **t=83.0s** |
| CHANGE_FALLING 开始 | **t=129.5s** |
| STABLE_NO_COIL 开始 | **t=131.5s** |
| **Coil 3** (用户核验) | |
| CHANGE_RISING 开始 | **t=145.0s** |
| STABLE_COIL 开始 | **t=148.0s** |
| CHANGE_FALLING 开始 | t=191.0s (用户) |
| STABLE_NO_COIL 开始 | **t=194.0s (用户)** |
| **Coil 4** | |
| CHANGE_RISING 开始 | t=223.0s |
| STABLE_COIL 开始 | **t=226.0s** |
| 视频结束 | t=274.0s |

## 时间分段表 (v4 完整视觉重判)

| 起 (s) | 止 (s) | 状态 |
|---:|---:|---|
| 0.0 | 10.5 | `STABLE_NO_COIL` |
| 10.5 | 13.5 | `CHANGE_RISING` |
| 13.5 | 61.5 | `STABLE_COIL` |
| 61.5 | 64.5 | `CHANGE_FALLING` |
| 64.5 | 79.5 | `STABLE_NO_COIL` |
| 79.5 | 83.0 | `CHANGE_RISING` |
| 83.0 | 129.5 | `STABLE_COIL` |
| 129.5 | 131.5 | `CHANGE_FALLING` |
| 131.5 | 145.0 | `STABLE_NO_COIL` |
| 145.0 | 148.0 | `CHANGE_RISING` |
| 148.0 | 191.0 | `STABLE_COIL` |
| 191.0 | 194.0 | `CHANGE_FALLING` |
| 194.0 | 223.0 | `STABLE_NO_COIL` |
| 223.0 | 226.0 | `CHANGE_RISING` |
| 226.0 | 274.0 | `STABLE_COIL` |


