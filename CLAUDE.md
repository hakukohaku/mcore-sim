# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mcore-sim is a SimPy-based discrete-event simulator for many-core architecture performance and power analysis. It models Computing-in-Memory (CIM) and DaVinci architectures with cores connected via mesh NoC and DRAM subsystems. It supports Tensor Parallel (TP) and Pipeline Parallel (PP) workloads and outputs cycle count, latency, and per-subsystem energy/power breakdowns as CSV.

## Commands

### Running Simulations

Simulation is a two-step process (instruction generation + simulation), automated by Make:

```bash
# Install dependencies
pip install -r requirements.txt

# Tensor Parallel (one-shot)
make run_tp BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7

# Pipeline Parallel (one-shot)
make run_pp BATCH=72 MICRO_BATCH=4 DP=2 ARCH=CIM TECH=7

# Generate instruction stream only (no simulation)
make gen_tp BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7

# Clean all outputs, logs, tests, data
make clean
```

Makefile variables: `BATCH`, `MICRO_BATCH`, `DP`, `PP`, `ARCH` (CIM or DAVINCI), `TECH` (7 or 22).

ARCH selects the matching config files automatically (arch/, config/, power/power_config/).

### Manual Two-Step Execution

```bash
# Step 1: Generate instruction stream
python tools/tp_inst_generation.py -b 1 -mb 1 -dp 2 -o tests/pipeline/batch_1_micro_1_dp_2.json -a arch/cim.json -c config/cim.json

# Step 2: Run simulator
python run.py -b 1 -mb 1 -dp 2 -t 7 --arch_name CIM --arch arch/cim.json --fail failslow/normal.json --workload tests/pipeline/batch_1_micro_1_dp_2.json --power power/power_config/cim_power_7nm.json --log log/sim.log --level debug --output output/results/result.csv
```

### Merging Results

```bash
python process_results.py --input_dir output/results --output output/all_results.csv
```

### No Formal Test Suite

There is no pytest or unittest suite. Validation is done via smoke tests (`make run_tp` / `make run_pp`).

## Architecture

### Simulation Flow

1. **Instruction generation** (`tools/tp_inst_generation.py` or `tools/pp_inst_generation.py`): reads arch config + model config, produces a JSON workload with per-PE instruction sequences
2. **Config parsing** (`run.py`): loads arch config (Pydantic via `ArchConfig`), workload, power params, fail-slow settings
3. **Architecture init** (`src/architecture.py` `Arch`): creates cores in a grid, connects them via mesh NoC, attaches DRAM controllers, builds the SimPy environment
4. **Simulation** (`Arch.run()`): each core runs a SimPy process executing its instruction stream; instructions decode into Task objects that execute on resources (TPU, Vector, LSU, Links)
5. **Output**: CSV with cycle count, latency, SRAM usage, and energy/power per subsystem (compute, DRAM, NoC, SRAM, CIM local read)

### Key Source Modules

- `src/sim_type.py` - Core data types: Task, ComputeTask, MemTask, CommunicationTask, Ring, Instruction, Workload. TaskType enum defines all operation codes.
- `src/architecture.py` - `Arch` class: top-level orchestrator. Creates cores, NoC, DRAM; runs simulation; collects results.
- `src/core.py` - `Core` class: contains TPU, Vector unit, LSU, SPMManager. Handles instruction decode, task execution, SPM allocation/free, remote DRAM via proxy tasks.
- `src/noc_new.py` - Mesh NoC with XY dimension-order routing. Router, Link classes. Bandwidth-limited message passing.
- `src/dram.py` - DRAM controller with bandwidth modeling.
- `src/arch_config.py` - Pydantic models for all JSON config validation.
- `src/common.py` - Global state: `power_summary` dict, `total_power`, `record_power_trace()`, `CFG` class, logging setup.

### Key Patterns

- **SimPy processes**: `yield env.timeout()`, `yield env.process()`, `yield resource.execute()`
- **Inter-core communication**: cores use `data_in`/`data_out` SimPy Store channels for SEND/RECV
- **Remote DRAM**: core generates `MemReqPayload`, sends via NoC, awaits `DerivativeRecv` response
- **Power tracking**: `record_power_trace()` accumulates energy into the global `power_summary` dict by category
- **RING collective**: simplified all-reduce latency model: `ceil(size_bytes / noc_link_width)` cycles, no explicit SEND/RECV

### Configuration Files

- `arch/*.json` - Hardware topology: core grid dimensions, compute FLOPS, SPM size, NoC mesh config, DRAM bandwidth/capacity, memory-to-core mapping
- `config/*.json` - Model structure (Transformer: n_blocks, dim, heads, ffn_dim, input_len) + parallelism settings (tp_config or pipeline_config with core_list)
- `power/power_config/*.json` - Per-operation energy costs (tpu_flop, vect_flop, dram_read/write, noc_hop, sram_read/write, cim_local_read) for 7nm and 22nm
- `failslow/normal.json` - Fault injection config (required arg but has no effect in normal mode)

### DP Note

Different DP values produce independent, identical hardware groups. The simulator only models one DP group; total system power = single group power x DP.
