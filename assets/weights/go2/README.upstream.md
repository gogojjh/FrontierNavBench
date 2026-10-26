# Go2 MoE-CTS v4.2(176k)盲走策略

导航仿真(`nav_sim_isaac/scripts/sim_sensor_publisher.py --robot go2`)用的 Go2 行走策略。
2026-10-13 用户定下：仿真里 Go2 只用这一份，旧的 SRU 自带平地策略和自训的 nav_loco_rl 线都不再用。
**只在仿真里用**，真机继续用宇树自带步态。

## 来源

| 项 | 值 |
|---|---|
| 上游仓库 | [wertyuilife2/go2_rl_robotlab](https://github.com/wertyuilife2/go2_rl_robotlab) @ `28b4516`(顶层子模块 `src/go2_rl_robotlab`) |
| 上游路径 | `deploy/pre_train/go2/go2_moe_cts_176k_0.6984.{pt,onnx}`,原样拷贝，没改过 |
| 算法 | MoE-CTS(混合专家 + 并发师生),盲走，不用高程图 |
| 训练 | Isaac Lab 2.3.2,176k 迭代；文件名里的 0.6984 是上游在 RoboGauge(MuJoCo)上的 benchmark 分 |
| 许可证 | Apache-2.0(上游 `LICENSE`),再分发时保留本说明里的出处 |

| 文件 | 大小 | sha256 |
|---|---|---|
| `go2_moe_cts_176k_0.6984.pt`(TorchScript,仿真用这份) | 6 398 610 B | `c602e749ac292921e3d6f5b2ab1749c4e4eaa6dbe51f751cce5b669b102d10a2` |
| `go2_moe_cts_176k_0.6984.onnx`(备用) | 6 374 215 B | `93ed62bcca5ddd969da1f35960619005f3d3ae692ff41cde79f20697e9b06641` |

`policy.yaml` 里也记了这两个 sha256。换权重要新建目录(`moe_cts_<版本>/`),不覆盖这份。

## 约定(详见 `policy.yaml`,代码在 `nav_loco/nav_loco/go2/mcts_policy.py`)

- **频率**:50 Hz(物理 200 Hz,每 4 步算一次);PD:kp 25 / kd 0.5。
- **关节顺序**:按腿排，依次 FL / FR / RL / RR,每条腿 hip / thigh / calf。Isaac 的 `dof_names` 按关节类型分组，
  所以要按名字映射(`JointMap`),名字对不上就报错。
- **观测**:每帧 45 维，各项先截到 ±100 再乘缩放，依次是
  机体系角速度 ×0.25、机体系重力方向、指令 (vx, vy, wz)、关节角 − 默认姿态、关节角速度 ×0.05、上一次动作。
  JIT 里自己存 10 帧历史(批大小只能是 1);从站姿切到走路时用同一帧连调 10 次填满(`prime`)。
- **动作**:12 维,target = 默认姿态 + 0.25 × a。默认姿态:hip ±0.1、thigh 前 0.8 / 后 1.0、calf −1.5。
- **指令范围**:训练里楼梯 / 障碍地形的上限是 vx ±1.0、vy ±1.0、wz ±1.5;导航仿真再和协议上限
  (`nav_sim.protocol`)取交集，生效范围是 vx ±1.0、vy ±0.6、wz ±1.5。
- **仿真整机**:`go2_odin.usd`(带 3 kg 实体负载，整机约 18.0 kg),关自碰撞，力矩上限髋 / 大腿 23.7、膝 45.43 N·m。
  和上游训练设置的差异见 `nav_loco/README.md`「与上游差异」。

## 仿真输入 / 输出

策略在台式机的 Isaac 发布器进程里跑，不经过 ROS。

| 方向 | 通道 | 内容 |
|---|---|---|
| 进 | ZMQ 控制端口(isaac-s1 是 6557),`cmd_vel` | vx / vy / wz,0.5 秒没新指令就停；Jetson 桥从 `/cmd_vel`(cmd_vel_gate 之后)转过来 |
| 出 | ZMQ 流端口(6556),`odom_batch` / `joint_state` / `sim_state` | 里程计、关节角、是否摔倒 |
| 出 | ZMQ 图像端口(6555) | 前视 RGB / 深度；`--video` 时图像端口 + 3 出跟拍画面 |

## 能力

评估结果在 `nav_loco/eval/go2_moe_cts_v4.2_176k/`(冒烟、RoboGauge、env1 台阶探针、SRU 闭环)。
各台阶高度的通过率、坡度、抗推等数字以那里的 README 为准，这里不重复抄。
