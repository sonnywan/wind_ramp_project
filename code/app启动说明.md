# 风电功率爬坡预警系统 · Streamlit 演示页面 启动说明

## 1. 依赖
- Python 3.12
- streamlit（已验证 1.66.1）、pandas、numpy、matplotlib
- 安装：`pip install streamlit`

## 2. 启动（在**项目根目录** `wind_ramp_project` 下执行）
```bash
streamlit run code/app.py
```
启动后浏览器打开终端提示的 Local URL（默认 http://localhost:8501）。

> 注意：`app.py` 内使用 `output/` 相对路径，**必须在项目根目录启动**，不要在 `code/` 子目录里启动。
> 若端口被占，可指定端口：`streamlit run code/app.py --server.port 8502`

## 3. 前置数据（已具备，无需重跑）
`app.py` 直接读取以下已有产物，**不重新训练**：
- `output/field_power.csv` —— 场站功率时序
- `output/ramp_events.csv` —— 真实爬坡事件
- `output/lgb_pred.npz` —— 三分类概率（cls 源）
- `output/reg_prob.npz` —— 回归残差校准概率（reg 源，创新点2 产物）
- `output/cost_curve.png`、`output/reliability_diagram.png` —— 结果图

## 4. 页面功能
### 侧栏配置
| 控件 | 说明 |
|---|---|
| 运行模式 | 离线回放 / 实时模拟器 |
| 概率源 | cls=三分类概率；reg=回归校准概率（切换即重新取数、重算告警） |
| 启用迟滞 | 开关；开=连续 2 步超阈触发、回落 0.8τ 撤销 |
| 阈值 τ | 0.05–0.95 滑动调节 |

### 离线回放
- 上：场站功率曲线 + 真实爬坡事件区间（红=上爬/绿=下爬）
- 下：上/下行概率曲线 + 阈值线 + 告警标记（▼上行 / ▲下行）
- 指标卡：上/下行真实事件数、命中率（简化覆盖统计，演示用）
- 展开区：代价曲线（创新点3）+ 可靠性图（创新点2）

### 实时模拟器
- 以 test 段真实模型输出的 p_up/p_dn 逐点滚动回放，模拟"实时来数 → 算概率 → 触发告警"
- 可调每帧推进步数、回放总步数，开始/暂停
- 实时显示当前 t_idx、功率、上下行概率、进度条、告警状态
- 硬件就绪后可替换为串口实时数据源

## 5. 与预警体系的对齐
- 事件定义、迟滞规则、阈值口径与 `code/run_event_eval.py` 完全一致（`HYST_ON_STEPS=2`、`HYST_OFF_RATIO=0.8`）
- 概率源切换对应创新点2（概率化校准）；迟滞 + 阈值 + 代价标定对应创新点3（代价敏感决策层）
- 完整事件级指标、代价标定、双源对照见 `output/event_eval_report.md`

## 6. 常见问题
- **页面报找不到文件**：确认在项目根目录启动，且 `output/` 下产物存在。
- **中文乱码**：脚本已设置 `Microsoft YaHei/SimHei` 字体与 `axes.unicode_minus=False`，Windows 一般正常。
- **实时模式不动**：需点开「▶ 开始/暂停」开关；每帧推进步数/回放总步数可调。
- **端口冲突**：加 `--server.port 8502` 换端口。

