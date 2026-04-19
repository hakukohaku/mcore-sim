# Many-Core Simulation
本项目是一个基于 SimPy 的架构性能与能耗模拟仿真器。聚焦于存算一体（CIM）架构，描述了一个由多个核心组成的系统（Many-Core System），核心之间通过片上网络（NoC）连接，并包含内存子系统。并支持通过参数配置模拟DaVinci等其他架构。它接收硬件架构配置、模型结构和工作负载作为输入，模拟模型在目标硬件上的执行过程，精确计算以下指标：
- 性能指标：latency、cycle
- 能耗指标：计算单元、DRAM、SPM、NoC 等组件的能量消耗与功率

最终，模拟结果以结构化格式输出至 CSV 文件，用于架构性能优化与能耗分析。

## Overview
[1.目录结构](#1目录结构)
[2.使用说明](#2get-started使用说明)
[3.常见问题](#3常见问题faq)
      
## 1.目录结构
```bash
mcore-sim 
├─ .vscode/                 # 调试配置文件                     
├─ arch/                    # 硬件架构配置文件(JSON)                            
├─ config/                  # 模型结构与流水线配置文件(JSON)               
├─ failslow/                # 故障注入文件（无需更改，对仿真无影响，仅兼容性要求）
├─ output/                  # 运行结果输出目录(CSV)
├─ power/                   # 操作能耗配置目录
├─ src                      # 核心模块目录
│  ├─ arch_config.py        # 架构配置Pydantic模型
│  ├─ architecture.py       # 架构核心逻辑
│  ├─ common.py             # 通用工具（CFG配置类、power_summary存储）
│  ├─ core.py               # 定义Core组件本身的行为
│  ├─ dram.py               # 定义DRAM组件本身的行为
│  ├─ draw.py               # 绘制仿真结果图表
│  ├─ noc_new.py            # 定义NoC组件本身的行为
│  ├─ sim_type.py           # 定义数据类型、任务类型、指令类型和通信机制
├─ analysis/                # 仿真后分析工具
├─ tests/                   # 工作负载存放目录
├─ tools/                   # 指令序列生成工具
├─ Makefile                 # 自动化执行脚本                        
├─ run.py                   # 主程序（单次仿真）
├─ run_all.py               # 批量并行仿真（多架构+多优化配置）
├─ process_results.py       # 将.csv合并
├─ README.md  
├─ requirements.txt         # 依赖库清单
```

## 2.Get Started/使用说明

### 2.1 Installation
Python 版本：推荐 Python 3.8~3.11（兼容 Pydantic、SimPy 等库，避免高版本兼容性问题）
```bash
pip install -r requirements.txt
```

### 2.2 批量并行仿真最终结果

使用 `run_all.py` 可一次性运行多种架构 × 优化配置的 encoder/decoder 仿真，**多进程并行**执行。

```bash
python run_all.py --tech 7 --batch 24 --micro_batch 1 --dp 2 \
    --config “DAVINCI,complex” \
    --config “CIM,complex” \
    --config “CIM,complex,complex_opt” \
    --config “CIM,complex,complex_opt,co_opt”
```

执行流程：
1. **Phase 1**（串行）：按架构（CIM/DAVINCI）生成 instruction JSON，相同架构的不同优化配置**共享同一份指令文件**，避免磁盘浪费
2. **Phase 2**（并行）：同时启动所有仿真进程（每个配置的 encoder + decoder），输出独立 CSV

上述命令会并行运行 8 个仿真（4 配置 × encoder/decoder），结果输出到 `output/results/`：
```
output/results/
├─ encoder_DAVINCI_complex.csv
├─ decoder_DAVINCI_complex.csv
├─ encoder_CIM_complex.csv
├─ decoder_CIM_complex.csv
├─ encoder_CIM_complex_complex_opt.csv
├─ decoder_CIM_complex_complex_opt.csv
├─ encoder_CIM_co_opt_complex_complex_opt.csv
└─ decoder_CIM_co_opt_complex_complex_opt.csv
```

#### run_all.py 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch` | 总批次大小 | 24 |
| `--micro_batch` | 微批次大小 | 1 |
| `--dp` | 数据并行度 | 2 |
| `--tech` | 工艺节点（7 或 22） | 7 |
| `--config` | 架构+优化标志，可多次指定 | （必选） |
| `--encoder_input_len` | Encoder 输入长度 | 68 |
| `--encoder_ekv` | Encoder KV 长度 | 0 |
| `--decoder_input_len` | Decoder 输入长度 | 64 |
| `--decoder_ekv` | Decoder KV 长度 | 68 |
| `--output_dir` | CSV 输出目录 | output/results |
| `--verbose` | 保留 log/power_trace/data 输出 | 默认关闭（quiet 模式，只生成 CSV） |

`--config` 格式为 `ARCH,flag1,flag2,...`，可用标志：

| 标志 | 说明 |
|------|------|
| `complex` | 复数模型 |
| `complex_opt` | 启用复数计算优化 |
| `co_opt` | 启用算芯协同优化 |
| `single_tp` | 单 TP 核模式，快速仿真，能耗按 TP 度缩放 |


### 2.3 单次仿真（Makefile）

注：由于不同DP会产生完全独立且同步工作的硬件组，我们仅仿真其中一个DP硬件组来加速仿真，DP的设置决定了该硬件组分配到的batch size以及最后系统的总功耗（单独硬件功耗×DP）

```bash

# Combined PP+TP
make run_pptp BATCH=24 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7

# 启用 quiet 模式（跳过 log/power_trace/data 输出，只保留 CSV）
make run_tp BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7 QUIET=1

# 启用复杂度缩放
make run_tp BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7 \
    COMPLEX_ENABLE=1 COMPLEX_OPT_ENABLE=1 CO_OPT_ENABLE=1

# 只生成指令序列（不运行仿真）
make gen_tp BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7
make gen_pptp BATCH=24 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7

# 只运行仿真（使用已生成的指令序列）
make sim_pptp BATCH=24 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7 \
    INST_STREAM=tests/pipeline/my_workload.json

# 清理所有输出
make clean
```

Makefile 变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BATCH` | 总批次大小 | 2 |
| `MICRO_BATCH` | 微批次大小 | 1 |
| `DP` | 数据并行度 | 2 |
| `PP` | 流水线级数 | 8 |
| `ARCH` | 架构（CIM / DAVINCI） | CIM |
| `TECH` | 工艺节点（7 / 22） | 7 |
| `QUIET` | 设为 1 跳过 log/trace 输出 | 空（不启用） |
| `COMPLEX_ENABLE` | 复杂度模式 | 空 |
| `COMPLEX_OPT_ENABLE` | 复杂度优化 | 空 |
| `CO_OPT_ENABLE` | 协同优化 | 空 |
| `INPUT_LEN` | 输入序列长度 | 空（读取 config） |
| `N_ENCODER_KV_LEN` | Encoder KV 长度 | 0 |
| `SINGLE_TP` | 单 TP 核模式 | 空 |
| `TP_SCALE` | 能耗缩放因子 | 空 |

### 2.4 分步手动运行

分步运行：生成指令序列 → 运行仿真。

步骤1：生成指令序列
```bash
# PP+TP 模式
python tools/pptp_inst_generation.py \
  -b 24 -mb 1 -dp 2 \
  -o tests/batch_24_micro_1_dp_2.json \
  -a arch/cim.json -c config/cim.json
```

步骤2：运行仿真
```bash
python run.py \
  -b 1 -mb 1 -dp 2 -t 7 \
  --arch_name CIM \
  --arch arch/cim.json \
  --fail failslow/normal.json \
  --workload tests/batch_1_micro_1_dp_2.json \
  --power power/power_config/cim_power_7nm.json \
  --log log/sim.log \
  --level debug \
  --output output/results/result.csv
```

run.py 额外参数：
- `--quiet`：跳过 log、power_trace、data trace 输出，只生成 CSV
- `--power_trace PATH`：指定 power trace 输出路径（quiet 模式下自动跳过）
- `--simstart` / `--simend`：指定仿真起止周期
- `--flow`：启用流追踪（输出到 `gen/`）
- `--tp_scale N`：能耗乘以 N（单 TP 核模式）
- `--complex_enable` / `--complex_opt_enable` / `--co_opt_enable`：复杂度缩放标志

### 2.5 合并结果

```bash
python process_results.py --input_dir output/results --output output/all_results.csv
```

## 3.常见问题(FAQ)

1. 如何自定义硬件架构？
复制 `arch/cim.json` 并修改参数，运行时通过 `--arch` 指定新文件：
```bash
python run.py --arch arch/my_custom.json
```

2. 磁盘空间不足？
batch 较大时 instruction JSON 文件很大（batch=24 约 1.2GB/文件）。`run_all.py` 会自动合并相同架构的指令文件以节省空间。也可用 `QUIET=1` 跳过 log/trace 输出（单次 batch=24 可节省数百 MB）。

3. 如何只看 CSV 结果？
使用 `run_all.py`（默认 quiet 模式）或 Makefile 加 `QUIET=1`，只生成 CSV，跳过 debug log、power trace、data trace 等中间文件。
