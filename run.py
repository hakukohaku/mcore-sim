import os
import sys
import csv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import time
import logging
import argparse
from src.arch_config import ArchConfig
from src.common import CFG
import src.common as common
from src.sim_type import Workload, FailSlow, PEworkload, Instruction
from src.architecture import Arch
from pydantic import ValidationError
from src.common import *

print("Python executable used:", sys.executable)
def fail_analyzer(filename: str) -> FailSlow:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            fail = FailSlow.model_validate(data)
            return fail
        except ValidationError as e:
            print(e.json())

def config_analyzer(filename: str) -> ArchConfig:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            config = ArchConfig.model_validate(data)
            return config
        except ValidationError as e:
            print(e.json())

def workload_analyzer(filename: str) -> Workload:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            workload = Workload.model_validate(data)
            return workload
        except ValidationError as e:
            print(e.json())

def setup_logging(filename, level):
    log_dir = os.path.dirname(filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level = level,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        filename = filename,
        filemode = 'w'
    )

def output_csv(filename: str):
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header=["Arch","Batch Size", "Micro Batch Size", "DP", "Cycle", "Latency", "E_Compute", "E_DRAM_Read", "E_DRAM_Write", \
            "E_DRAM_Access", "E_NOC_Hop", "E_SRAM_Read", "E_SRAM_Write", "E_SRAM_Access","E_CIM_Local_Read", "Total_Energy","P_Compute", "P_DRAM_Read", "P_DRAM_Write", \
            "P_DRAM_Access", "P_NOC_Hop", "P_SRAM_Read", "P_SRAM_Write", "P_SRAM_Access","P_CIM_Local_Read", "P_Total_Power"]
        writer.writerow(header)
        return writer

def main():
    print("Reading architecture config.")
    arch_config = config_analyzer(args.arch)
    print("Finished.")

    print("Loading fail-slow setting.")
    fail_slow = fail_analyzer(args.fail)
    print(f"Finished loading fail-slow setting.")

    print("Loading simulation workload.")
    workload = workload_analyzer(args.workload)
    print(f"Finished loading {workload.name} as simulation workload.")

    print("Setting up logging.")

    if args.level == "debug":
        level = logging.DEBUG
    elif args.level == "info":
        level = logging.INFO
    
    setup_logging(args.log, level)

    print("Finished.")

    stage = None
    arch = Arch(arch_config, [pe.insts for pe in workload.pes], fail_slow, workload.name, args.fail, args.power, stage)

    start_time = time.time()
    result = arch.run()
    end_time = time.time()
    simulation_time = end_time - start_time

    
    output_csv(args.output)

    if stage == "pre_analysis":
        stage = "post_analysis"
        arch = Arch(arch_config, [pe.insts for pe in workload.pes], fail_slow, stage)
        result = arch.run()
        end_time = time.time()
        simulation_time = end_time - start_time

    freq = arch_config.freq
    arch_name = args.arch_name
    batchsize = args.batch
    micro_batch_size = args.micro_batch
    dp = args.dp
    cycle = arch.end_time
    latency = arch.end_time/1e6/freq
    e_compute = common.power_summary["compute"]
    e_dram_read = common.power_summary["dram_read"]
    e_dram_write = common.power_summary["dram_write"]
    e_dram_access = common.power_summary["dram_read"] + common.power_summary["dram_write"]
    e_noc_hop = common.power_summary["noc_hop"]
    e_sram_read = common.power_summary["sram_read"] if arch_name == "CIM" else (common.power_summary["sram_read"] + common.power_summary["cim_local_read"])
    e_sram_write = common.power_summary["sram_write"]
    e_sram_access = e_sram_read + common.power_summary["sram_write"]
    e_cim_local_read = common.power_summary["cim_local_read"] if arch_name == "CIM" else 0
    total_energy = common.total_power
    p_compute = dp * e_compute / (cycle/freq)
    p_dram_read = dp * e_dram_read / (cycle/freq)
    p_dram_write = dp * e_dram_write / (cycle/freq)
    p_dram_access = dp * e_dram_access / (cycle/freq)
    p_noc_hop = dp * e_noc_hop / (cycle/freq)  
    p_sram_read = dp * e_sram_read / (cycle/freq)
    p_sram_write = dp * e_sram_write / (cycle/freq)
    p_sram_access = dp * e_sram_access / (cycle/freq)
    p_cim_local_read = dp * e_cim_local_read / (cycle/freq)
    p_total_power = dp * common.total_power / (cycle/freq)
    
    result_data = [arch_name, batchsize, micro_batch_size, dp, cycle, latency, e_compute, e_dram_read, e_dram_write, e_dram_access, e_noc_hop, \
            e_sram_read, e_sram_write, e_sram_access, e_cim_local_read, total_energy, p_compute, p_dram_read, p_dram_write, p_dram_access, \
            p_noc_hop, p_sram_read, p_sram_write, p_sram_access, p_cim_local_read, p_total_power]
    print("Writing result data...")
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(result_data)

    # arch.debug()

    print("="*40)
    print(f"Simulation time is {simulation_time}.")
    print(f"Total simulation cycles is {arch.end_time:.0f}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #set simulation start and end cycle
    parser.add_argument("--simstart", type=int, default=0,
                        help="Simulation start cycle, default is 0")
    parser.add_argument("--simend", type=int, default=int((1<<31)-1),
                        help="Simulation end cycle, default is None (natural end)")
    parser.add_argument("--flow", action="store_true", help="enable flow flag")
    parser.add_argument("--arch_name", type=str, default="CIM")
    parser.add_argument("--workload", type=str, default="tests/pipeline/workload_pipeline_fixed_test.json")
    parser.add_argument("--arch", type=str, default="arch/gemini4_4.json")
    parser.add_argument("--fail", type=str, default="failslow/normal.json")
    parser.add_argument("--power", type=str, default="power/power_config/power.json")
    parser.add_argument("--log", type=str, default="logging/log.txt")
    parser.add_argument("--level", type=str, default="info")
    parser.add_argument("--output", type=str, default="output.csv")
    parser.add_argument('-b', '--batch', type=int,help='batch size')
    parser.add_argument('-mb', '--micro_batch', type=int, help='micro batch size')
    parser.add_argument('-dp', '--dp', type=int, help='data parallelism')

    args = parser.parse_args()
    common.cfg=common.CFG(args)
    main()
