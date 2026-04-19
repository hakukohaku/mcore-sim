# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mcore-sim is a SimPy-based discrete-event simulator for many-core architecture performance and power analysis. It models Computing-in-Memory (CIM) and DaVinci architectures with cores connected via mesh NoC and DRAM subsystems. It supports Tensor Parallel (TP), Pipeline Parallel (PP), and combined PP+TP workloads, outputting cycle count, latency, and per-subsystem energy/power breakdowns as CSV.

## Commands

### Running Simulations

Simulation is a two-step process (instruction generation + simulation), automated by Make:

```bash
pip install -r requirements.txt   # Python 3.8–3.11 recommended

# Tensor Parallel
make run_tp BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7

# Pipeline Parallel
make run_pp BATCH=72 MICRO_BATCH=4 DP=2 ARCH=CIM TECH=7

# Combined PP+TP
make run_pptp BATCH=24 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7

# Generate instruction stream only (no simulation)
make gen_tp BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7

# Clean all outputs, logs, tests, data
make clean
```

Makefile variables: `BATCH`, `MICRO_BATCH`, `DP`, `PP`, `N_ENCODER_KV_LEN`, `ARCH` (CIM or DAVINCI), `TECH` (7 or 22). ARCH selects matching config files automatically (arch/, config/, power/power_config/). Additional optional variables: `INPUT_LEN`, `SINGLE_TP` (flag), `TP_SCALE`, `COMPLEX_ENABLE`, `COMPLEX_OPT_ENABLE`, `CO_OPT_ENABLE`.

### Manual Two-Step Execution

```bash
# Step 1: Generate instruction stream
python tools/tp_inst_generation.py -b 1 -mb 1 -dp 2 -o tests/pipeline/batch_1_micro_1_dp_2.json -a arch/cim.json -c config/cim.json

# Step 2: Run simulator
python run.py -b 1 -mb 1 -dp 2 -t 7 --arch_name CIM --arch arch/cim.json --fail failslow/normal.json --workload tests/pipeline/batch_1_micro_1_dp_2.json --power power/power_config/cim_power_7nm.json --log log/sim.log --level debug --output output/results/result.csv
```

Additional run.py flags: `--simstart`/`--simend` (cycle range), `--flow` (enable flow tracing to `gen/`), `--tp_scale N` (multiply energy by N for single-TP-core mode), `--complex_enable`, `--complex_opt_enable`, `--co_opt_enable` (see Complexity Scaling Flags below).

### Merging Results

```bash
python process_results.py --input_dir output/results --output output/all_results.csv
```

### No Formal Test Suite

There is no pytest or unittest suite. Validation is done via smoke tests (`make run_tp` / `make run_pp`). Stdout is redirected to `output/run.log` during Make runs.

## Architecture

### Simulation Flow

1. **Instruction generation** (`tools/tp_inst_generation.py`, `tools/pp_inst_generation.py`, or combined PP+TP in `tp_inst_generation.py` with `pptp_config`): reads arch + model config, produces JSON workload with per-PE instruction sequences
2. **Config parsing** (`run.py`): loads arch config (Pydantic `ArchConfig`), workload, power params, fail-slow settings into global `common.cfg`
3. **Architecture init** (`src/architecture.py` `Arch`): creates cores in a grid, connects via mesh NoC, attaches DRAM controllers, builds the SimPy environment
4. **Simulation** (`Arch.run()`): each core runs a SimPy process executing its instruction stream; instructions decode into Task objects that execute on resources
5. **Output**: CSV with 26 columns covering cycles, latency, SRAM usage, and energy/power per subsystem. Power trace written to `power/power_trace/power_trace.txt`. Analysis trace data (comp_trace.json, comm_trace.json) written to `data/<net_name>/<fail_kind>/`

### Key Source Modules

- `src/sim_type.py` — Core data types: Task hierarchy, TaskType enum, Instruction, Workload, DataType (FEAT=activations, PARA=weights)
- `src/architecture.py` — `Arch` class: top-level orchestrator. Creates cores, NoC, DRAM; runs simulation; collects results
- `src/core.py` — `Core` class: contains TPU, Vector unit, LSU, SPMManager, TableScheduler. Handles instruction decode, task execution, SPM allocation/free, remote DRAM via proxy tasks
- `src/noc_new.py` — Mesh NoC with XY/YX/balanced dimension-order routing. Router, Link, NoC classes. Supports hop-by-hop and wormhole modes
- `src/dram.py` — DRAM controller with bandwidth modeling (serialized access via MonitoredResource)
- `src/arch_config.py` — Pydantic models for all JSON config validation
- `src/common.py` — Global state: `power_summary` dict, `total_power`, `record_power_trace()`, `CFG` class, `MonitoredResource`
- `analysis/` — Post-simulation trace analysis: `comm_fail.py` builds PE-to-PE dependency graphs, `comp_fail.py` analyzes compute bottlenecks, `trace_format.py` defines compressed trace Pydantic models

### TaskType Enum

```
READ=0, WRITE=1, SEND=2, RECV=3, STAY=4, RING=16
CONV=5, POOL=6, FC=7, ELEM=8, GCONV=9, PTP=10, TRANS=11
LOAD=12, STORE=13, FREE=14, NON_LINEAR=15
```

Task hierarchy: `Task` base → `ComputeTask` (CONV/FC/ELEM/etc), `IOTask` (READ/WRITE), `MemTask` (READ/WRITE to DRAM with local/remote routing), `CommunicationTask` (SEND/RECV/RING), Memory management tasks (LOAD/STORE/FREE). `DerivativeTask` mixin marks runtime-generated proxy tasks (DerivativeSend, DerivativeRecv for remote DRAM access).

### Instruction Trigger/Dependency System

- `trigger_index[]`: instruction indices this instruction triggers upon completion
- `trigger_core_id[]`: which core owns each triggered instruction (enables cross-core WRITE triggers)
- `feat_num` / `para_num`: number of expected inputs before an instruction is ready to execute
- When a task completes, `TableScheduler.task_update()` decrements operand counts on triggered instructions; once all inputs arrive, the instruction enters the waiting queue

### Block-Based Scheduling

The scheduler (`TableScheduler`) divides each core's program into blocks of `blk_size` instructions (from `arch.core.blk_size` in arch config). Only the current block's instructions are active. When all instructions in a block complete, the scheduler advances to the next block via `task_block_update()`. Messages for future blocks are buffered in `recv_queue`.

### SimPy Resource Model

- **MonitoredResource**: wraps SimPy Resource for bandwidth-limited hardware (TPU, LSU, DRAM). `exe(task, delay)` acquires, waits delay cycles, releases. Contention causes natural SimPy blocking
- **SPMManager.container**: SimPy Container for SRAM allocation/deallocation with blocking when full. Tracks per-core and total SRAM usage, reports peaks
- **Link/Router stores**: PriorityStore for NoC message queues

### Inter-Core Communication

- **Point-to-point**: SEND puts message on `data_out` Store → Router → mesh routing → destination Router → `data_in` Store → RECV. Supports path broadcasting via `path_dst` field
- **RING collective**: simplified all-reduce latency model: `size_bytes / noc_link_width` cycles, power charged as `noc_hop`. No explicit SEND/RECV generated. Zero-size payload is a no-op
- **Remote DRAM access**: core creates `MemReqPayload` → sends MEM_REQUEST via NoC → memory-attached core creates proxy Read task → executes on DRAM → DerivativeSend returns data via NoC → requester's DerivativeRecv completes. This is transparent to the instruction stream

### Power Tracking

`record_power_trace()` in `common.py` accumulates energy into `power_summary` dict by category:
- **compute**: `flops * tpu_flop_power` or `flops * vect_flop_power` (NonLinear uses vect unit)
- **dram_read/write**: `bytes * dram_read_power` (per byte)
- **noc_hop**: `hops * noc_hop_power` (per 64-byte hop)
- **sram_read/write**: `bytes * sram_read_power` (per byte)
- **cim_local_read**: parameter reads in CIM architecture

Final power = `energy * dp / (cycles / freq)`. Power configs are in `power/power_config/` for 7nm and 22nm.

CIM vs DaVinci power difference: CIM reports `cim_local_read` separately; DaVinci folds it into `sram_read` (see `run.py` output logic).

### Complexity Scaling Flags

Three CLI flags (also available as Makefile variables) modify compute and data scaling at simulation time. They only affect `run.py`; instruction generation is unchanged.

```bash
# Example: CIM with all optimizations enabled
make run_tp BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7 \
    COMPLEX_ENABLE=1 COMPLEX_OPT_ENABLE=1 CO_OPT_ENABLE=1
```

| Flag | Effect |
|------|--------|
| `--complex_enable` | FLOPs ×4, all data transfer/access sizes ×2 (DRAM, NoC, SRAM, Ring) |
| `--complex_opt_enable` | Compute delay ×0.75, compute power ×0.83 |
| `--co_opt_enable` | cim_local_read power ×0.73, compute power ×0.77 |

When `--complex_opt_enable` and `--co_opt_enable` are both enabled, compute power uses the additive reduction: ×0.6 (i.e. 1 − 0.17 − 0.23), not the multiplicative 0.83 × 0.77.

**Implementation**: `common.py` stores five scaling globals (`data_scale`, `flops_scale`, `compute_delay_factor`, `compute_power_factor`, `cim_power_factor`) initialized by `init_scaling_factors()`. These are applied in `ComputeTask.run()`, `MemTask.size_in_bytes()`, `MemTask.run()`, `Send.run()`, `Ring.run()` (in `sim_type.py`) and `Link.calc_latency()`, `process_hop_by_hop()`, `wormhole_send()` (in `noc_new.py`).

### TP vs PP Instruction Generation

| Aspect | TP | PP |
|--------|----|----|
| Parallelization | Splits K dimension across cores | Splits model layers across pipeline stages |
| Communication | RING collective (AllReduce) after compute | Point-to-point SEND/RECV between stages |
| First input | READ from DRAM per core | READ only on first core; others RECV |
| Output | STORE locally | SEND to next stage or STORE |
| Tool | `tools/tp_inst_generation.py` | `tools/pp_inst_generation.py` |

PP+TP mode uses `pptp_config` in config JSON, which specifies `pp_core_groups` (list of per-stage core lists) and `tp` degree within each stage. Generated by `tools/tp_inst_generation.py` reading `pptp_config`.

### DP Note

Different DP values produce independent, identical hardware groups. The simulator only models one DP group; total system power = single group power × DP.

### Configuration Files

- `arch/*.json` — Hardware topology: core grid (x,y), compute FLOPS (tpu + vect), SPM size, `blk_size` (scheduler block size), NoC mesh config (routing type XY/YX, link width/delay), DRAM bandwidth/capacity, memory-to-core mapping (`all_core_distributed` or `two_sides_edge` with explicit `mem_core` map), `freq` (GHz clock frequency)
- `config/*.json` — Model structure (Transformer: n_blocks, dim, heads, ffn_dim, input_len) + parallelism settings (`tp_config`, `pipeline_config`, `pptp_config`). Also contains `freq` (must match arch config)
- `power/power_config/*.json` — Per-operation energy costs (per-flop, per-byte, per-hop) for 7nm and 22nm. 8 fields: tpu_flop_power, vect_flop_power, dram_read_power, dram_write_power, noc_hop_power, sram_write_power, sram_read_power, cim_local_read_power
- `failslow/normal.json` — Fault injection config (required arg, no effect in normal mode: all lists empty)
