# Many-Core Simulation
本项目是一个基于 SimPy 的架构性能与能耗模拟仿真器。聚焦于存算一体（CIM）架构，描述了一个由多个核心组成的系统（Many-Core System），核心之间通过片上网络（NoC）连接，并包含内存子系统。并支持通过参数配置模拟DaVinci等其他架构。它接收硬件架构配置、模型结构和工作负载作为输入，模拟模型在目标硬件上的执行过程，精确计算以下指标：
- 性能指标：latency、cycle
- 能耗指标：计算单元、DRAM、SPM、NoC 等组件的能量消耗与功率
最终，模拟结果以结构化格式输出至 CSV 文件，用于架构性能优化与能耗分析。

## Overview
[1.目录结构](#1目录结构)
[2.使用方法](#2get-started使用说明)
[3.配置文件详解](#3配置文件详解)
[4.常见问题](#4常见问题faq)
示例和截图（无）
      
## 1.目录结构
```bash
mcore-sim                                       
├─ arch/                    # 硬件架构配置文件(JSON)                            
├─ config/                  # 模型结构与流水线配置文件(JSON)               
├─ data/                    # 运行时指令真实执行顺序
├─ log/                     # 日志输出目录
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
├─ README.md                                                               
├─ pp_results.sh            # 参数化扫描仿真脚本
├─ proc.sh                                           
├─ process_results.py       # 将.csv合并
├─ requirements.txt         # 依赖库清单
├─ run.py                   # 主程序
```


## 2.Get Started/使用说明

### 2.1 Installation
Please run the following commands to create a python environment and install required packages.

Python 版本：推荐 Python 3.8~3.11（兼容 Pydantic、SimPy 等库，避免高版本兼容性问题）
```bash
pip install -r requirements.txt
```
### 2.2 运行仿真（三种方式）
#### 方式一：使用Makefile一键运行（快速测试默认配置）
```bash
# 1. 克隆项目
git clone -b project_wc https://github.com/hakukohaku/mcore-sim.git
cd mcore-sim

# 2. 一键运行仿真（自动调用默认配置）
make run
```
说明：
- Makefile 默认配置：CIM 架构、固定 batch/micro_batch/dp 参数。
- 若需修改默认参数，直接编辑项目根目录的 Makefile，修改 BATCH、MICRO_BATCH、DP 等变量值。

#### 方式二：参数化扫描
适合需要测试 “不同 batch/micro_batch/ 架构” 对性能 / 能耗影响的场景，通过脚本循环扫描参数：
```bash
# 扫描参数BATCH、MICRO_BATCH、ARCH
source pp_result.sh
```
输出结果​​：多组配置的合并结果文件，可分析规律。

#### 方式三：自定义运行
分两步：生成指令序列 → 运行仿真。
- 步骤1：生成指令序列(tools/test_pp.py)
```bash
# 命令格式
python tools/test_pp.py \
  -b [BATCH] \          # 总批次大小（必选，如 8）
  -mb [MICRO_BATCH] \   # 微批次大小（必选，如 2）
  -dp [DP] \            # 数据并行度（必选，如 4）
  -o [INST_STREAM] \    # 指令序列输出路径（必选，如 data/inst_stream.json）
  -a [ARCH_FILE] \      # 硬件架构配置文件（必选，如 arch/cim.json）
  -c [CONFIG_FILE]      # 模型配置文件（必选，如 config/cim.json）

# 示例：生成 CIM 架构、batch=8、micro_batch=2 的指令序列
python tools/test_pp.py \
  -b 8 \
  -mb 2 \
  -dp 4 \
  -o data/cim_inst_8_2.json \
  -a arch/cim.json \
  -c config/cim.json
```
- 步骤2：运行仿真(run.py)
使用生成的指令序列，配置能耗、日志等参数，执行仿真。
```bash
# 命令格式
python run.py \
  -b [BATCH] \          # 总批次大小（必选，需与步骤 1 一致，如 8）
  -mb [MICRO_BATCH] \   # 微批次大小（必选，需与步骤 1 一致，如 2）
  -dp [DP] \            # 数据并行度（必选，需与步骤 1 一致，如 4）
  --arch_name [ARCH] \  # 架构名（可选，默认 CIM，如 CIM/DaVinci）
  --arch [ARCH_FILE] \  # 硬件架构配置文件（必选，如 arch/cim.json）
  --workload [INST_STREAM] \  # 指令序列路径（必选，步骤 1 生成的文件，如 data/cim_inst_8_2.json）
  --power [POWER_FILE] \      # 功率配置文件（必选，如 power/power_config/cim_power.json）
  --log [LOG_FILE] \    # 日志输出路径（可选，默认 log/run.log）
  --level [LOG_LEVEL] \ # 日志级别（可选，info/debug，默认 info）
  --output [OUTPUT_CSV] # 结果输出路径（可选，默认 output/result_8_2.csv）
  > [RUN_LOG] 2>&1      # 重定向控制台输出到日志（可选，如 log/console.log）

# 示例：运行 CIM 架构仿真，输出日志与结果
python run.py \
  -b 8 \
  -mb 2 \
  -dp 4 \
  --arch_name CIM \
  --arch arch/cim.json \
  --workload data/cim_inst_8_2.json \
  --power power/power_config/cim_power.json \
  --log log/cim_sim.log \
  --level debug \
  --output output/cim_result_8_2.csv \
  > log/console.log 2>&1
```

## 3.配置文件详解
所有配置文件均为 JSON 格式，需严格匹配字段要求，Pydantic 会自动验证，格式错误会报错。
### （1）硬件架构配置(../arch/cim.json)
```bash
{
	"freq" :0.8,                    # 系统主频(GHz)
	"core" : {
		"type" : "Simple",	# 核心类型
		"x" : 5,                    # 总核心数5*2=10
		"y" : 2,
		"width" : 8,                # 核心数据位宽(bit)
		"blk_size" : 100000000,		# 数据传输块大小(byte)
		"spm" : {
			"size" : 209715200,     # spm的容量(byte,200MB)
			"delay" : 0             # spm的访问延迟
		},
		"compute" : {
			"flops" : 1024,         # 标量浮点运算单元的计算能力
			"vect_flops" : 32       # 向量浮点运算单元的计算能力
		},
		"lsu" : {
			"width" : 4             # 访存单元带宽(byte/period)
		}
	},
	"noc" : {
		"type" : "Mesh",            # 网格拓扑
		"x" : 5,                    # 与核心数对应
		"y" : 2,
		"router" : {
			"type" : "XY",          # XY路由算法
			"vc" : 2                # 虚拟通道数量
		},
		"link" : {
			"width" : 16,           # 链路位宽(bit)
			"delay" : 0             # 链路传输延迟
		}
	},
	"mem" : {
		"type" : "two_sides_edge",  # 双端DRAM控制器？？
		"num" : 1,                  # DRAM控制器数量
		"dram_bw" : 128,            # DRAM的峰值带宽(GB/s)
		"dram_capacity" : 1048576,  # DRAM总容量(byte,1MB)
		"mem_core" : {
			"0": {"dram_id": 0}  # 核心0（坐标(0,0)）连接DRAM控制器0
		}
	}
}
```
### （2）模型结构配置(../config/cim.json)
```bash
{
	"model" : {
		"n_channel" : 16,       # 每个样本被分割成的块数？？
		"n_layers" : 10,        # 解码器/译码器堆叠层数
		"n_head" : 1,           # 注意力头数
		"n_kv_head" : 1,        # 键值头数
		"dim" : 32,             # 模型内部维度
		"head_dim" : 32,        # 注意力头维度，总维度 = n_head × head_dim
		"ffn_dim" : 64,         # 前馈神经网络中间层维度
		"input_len" : 48        # 输入序列长度（token数）
	},
	"pipeline_config" : {
		"core_list" : [0, 1, 2, 3, 4, 9, 8, 7, 6, 5],  # 流水线核心执行顺序
		"loop" : 1              # 流水线循环次数
	}
}
```
### （3）CIM架构下能耗配置(../power/power_config/cim_power.json)
```bash
{
  "tpu_flop_power": 0.000097,      # 标量浮点运算能耗
  "vect_flop_power": 0.000170,     # 向量浮点运算能耗
  "dram_read_power": 0.032,        # DRAM读能耗
  "dram_write_power": 0.032,       # DRAM写能耗
  "noc_hop_power": 0.001784,       # NoC每跳能耗
  "sram_write_power": 0.001568,    # SRAM写能耗
  "sram_read_power": 0.001568,     # SRAM读能耗
  "cim_local_read_power": 0.000436 # CIM本地读能耗
}
```

## 4.常见问题(FAQ)

1. 如何自定义硬件架构？
复制 arch/cim.json并修改参数，运行时通过 --arch指定新文件：
python run.py --arch arch/my_custom.json



