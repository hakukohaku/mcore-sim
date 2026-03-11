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
├─ tests/                   # 工作负载存放目录
├─ tools/                   # 指令序列生成工具
├─ Makefile                 # 自动化执行脚本                        
├─ process_results.py       # 将.csv合并
├─ README.md  
├─ requirements.txt         # 依赖库清单
├─ results_gen.sh           # 参数化扫描仿真脚本
├─ run.py                   # 主程序
```

## 2.Get Started/使用说明

### 2.1 Installation
Please run the following commands to create a python environment and install required packages.

Python 版本：推荐 Python 3.8~3.11（兼容 Pydantic、SimPy 等库，避免高版本兼容性问题）
```bash
pip install -r requirements.txt
```
### 2.2 运行仿真（TP / PP 两种方式）

注：由于不同DP会产生完全独立且同步工作的硬件组，我们仅仿真其中一个DP硬件组来加速仿真，DP的设置决定了该硬件组分配到的batch size以及最后系统的总功耗（单独硬件功耗×DP）

#### 方式一：使用 Makefile 一键运行

当前仓库按两种并行映射方式分别提供入口：
- `make run_tp`：Tensor Parallel，读取 `config/*.json` 中的 `tp_config`
- `make run_pp`：Pipeline Parallel，读取 `config/*.json` 中的 `pipeline_config`

模型配置统一读取 `model` 字段，其中层数字段为 `n_blocks`。

```bash
# Tensor Parallel
make run_tp BATCH=72 MICRO_BATCH=4 DP=2 ARCH=CIM TECH=7

# Pipeline Parallel
make run_pp BATCH=72 MICRO_BATCH=4 DP=2 ARCH=CIM TECH=7

#BATCH=[BATCH] \               # 总批次大小
#MICRO_BATCH=[MICRO_BATCH] \   # 微批次大小
#DP=[DP] \                     # 数据并行度
#ARCH=[CIM/DAVICIN]            # 硬件架构
#TECH=7/22                     # 工艺节点
```

说明：
- 若需修改默认参数，直接编辑项目根目录的 Makefile，修改 BATCH、MICRO_BATCH、DP 等变量值。

#### 方式二：分步运行

分步运行：生成指令序列 → 运行仿真。
- 步骤1：生成指令序列

TP 模式：
```bash
python tools/tp_inst_generation.py \
  -b [BATCH] \          # 总批次大小（必选，如 8）
  -mb [MICRO_BATCH] \   # 微批次大小（必选，如 2）
  -dp [DP] \            # 数据并行度（必选，如 4）
  -o [INST_STREAM] \    # 指令序列输出路径（必选，如 tests/batch_1_micro_1_dp_2.json）
  -a [ARCH_FILE] \      # 硬件架构配置文件（必选，如 arch/cim.json）
  -c [CONFIG_FILE]      # 模型配置文件（必选，如 config/cim.json）

# 示例：生成 TP 指令序列
python tools/tp_inst_generation.py \
  -b 1 \
  -mb 1 \
  -dp 2 \
  -o tests/batch_1_micro_1_dp_2.json \
  -a arch/cim.json \
  -c config/cim.json
```

PP 模式：
```bash
python tools/pp_inst_generation.py \
  -b 1 \
  -mb 1 \
  -dp 2 \
  -o tests/batch_1_micro_1_dp_2.json \
  -a arch/cim.json \
  -c config/cim.json
```

配置约定：
- `tools/tp_inst_generation.py` 只读取 `tp_config`
- `tools/pp_inst_generation.py` 只读取 `pipeline_config`
- `model.n_blocks` 表示 Transformer block 数量

- 步骤2：运行仿真(run.py)
使用生成的指令序列，配置能耗、日志等参数，执行仿真。
```bash
# 命令格式
python run.py \
  -b [BATCH] \          # 总批次大小（必选，需与步骤 1 一致）
  -mb [MICRO_BATCH] \   # 微批次大小（必选，需与步骤 1 一致）
  -dp [DP] \            # 数据并行度（必选，需与步骤 1 一致）
  -t [Tech] \           # 工艺节点，仅可选 7/22
  --arch_name [ARCH] \  # 架构名（可选，默认 CIM，如 CIM/DaVinci）
  --arch [ARCH_FILE] \  # 硬件架构配置文件（必选，如 arch/cim.json）
  --workload [INST_STREAM] \  # 指令序列路径（必选，步骤 1 生成的文件，如 tests/batch_1_micro_1_dp_2.json）
  --power [POWER_FILE] \      # 功率配置文件（必选，如 power/power_config/cim_power_7nm.json）
  --log [LOG_FILE] \    # 日志输出路径（可选，默认 log/run.log）
  --level [LOG_LEVEL] \ # 日志级别（可选，info/debug，默认 info）
  --output [OUTPUT_CSV] # 结果输出路径（可选，默认 output/results/results_arch_CIM_batch_1_micro_1_dp_2_tech_7nm.csv）

# 示例：运行 CIM 架构仿真，输出日志与结果
python run.py \
  -b 1 \
  -mb 1 \
  -dp 2 \
  -t 7  \
  --arch_name CIM \
  --arch arch/cim.json \
  --workload tests/batch_1_micro_1_dp_2.json \
  --power power/power_config/cim_power_7nm.json \
  --log log/cim_sim.log \
  --level debug \
  --output output/results/results_arch_CIM_batch_1_micro_1_dp_2_tech_7nm.csv \
```

### 2.3 `RING` 指令（简化 Ring All-Reduce）
`RING` 用于近似模拟 ring all-reduce 的通信开销，不需要指定 `src/dst`，不会显式生成 `SEND/RECV` 指令。

- `inst_type`: `16`（`TaskType.RING`）
- `tensor_slice`: 用于按数据量与 `NoC` 带宽自动估算延时和功耗（可保留与当前层输出一致的切片）
- `feat_num`: 若 `RING` 依赖上一条产出指令，通常设为 `1`
- `position/path_dst`: 对 `RING` 无效，可省略

使用约定：
- `RING` 是本地延时建模，不做 collective 同步，也不建模 NoC 拥堵。
- `RING` 延时固定按 `ceil(size_in_bytes, noc.link.width)` 自动估算，不再支持手动指定固定 cycle。
- 当 `tensor_slice` 对应的数据量为 `0` 时，`RING` 视为 no-op：`cycle = 0`，功耗也为 `0`。
- 当数据量大于 `0` 时，通信功耗直接计入 `noc_hop`，不会单独统计一个 `ring` 功耗项。
- 若要表示“上一层输出后再做一次 all-reduce 延时”，请让上一条指令通过 `trigger_index` 触发该 `RING`，并将 `RING.feat_num` 设为 `1`。

示例：
```json
{
  "inst_type": 16,
  "index": 10001,
  "trigger_index": [],
  "trigger_core_id": [],
  "layer_id": 0,
  "data_type": 1,
  "tensor_slice": [
    { "start": 0, "end": 4096 }
  ],
  "feat_num": 1,
  "para_num": 0
}
```

## 3.常见问题(FAQ)

1. 如何自定义硬件架构？
复制 arch/cim.json并修改参数，运行时通过 --arch指定新文件：
python run.py --arch arch/my_custom.json
