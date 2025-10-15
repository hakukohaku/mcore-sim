import simpy
#monitor resource such as lsu and tpu
import json
import time
import logging
import argparse
from src.sim_type import Instruction
from typing import List, Dict
from pydantic import ValidationError

logger = logging.getLogger("Common")

class CFG:
    def __init__(self, args):
        self.simstart = args.simstart
        self.simend = args.simend
        self.flow = args.flow

# parser = argparse.ArgumentParser()

# #set simulation start and end cycle
# parser.add_argument("--simstart", type=int, default=0,
#                     help="Simulation start cycle, default is 0")
# parser.add_argument("--simend", type=int, default=int((1<<31)-1),
#                     help="Simulation end cycle, default is None (natural end)")
# parser.add_argument("--flow", action="store_true", help="enable flow flag")
# parser.add_argument("--workload", type=str, default="tests/pipeline/workload_pipeline_fixed_test.json")
# parser.add_argument("--arch", type=str, default="arch/gemini4_4.json")
# parser.add_argument("--fail", type=str, default="failslow/normal.json")
# parser.add_argument("--log", type=str, default="logging/log.txt")
# parser.add_argument("--level", type=str, default="info")

# args = parser.parse_args()
# cfg = CFG(args)
# def build_cfg(cli_args=None):
#     ns = cli_args if cli_args is not None else parser.parse_args()
#     return CFG(ns)

# 占位，运行入口来注入
cfg = None

#record instruction dependency among cores
cores_deps=[]


class MonitoredResource(simpy.Resource):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []
        
    def checkneed(self):
        if self._env.now >= cfg.simstart and self._env.now <= cfg.simend:
            return True
        else:
            return False

    def exe(self, task, delay, ins, v=None, attributes=None):
        logger.debug(f"Time {self._env.now:.2f}: Task '{task}' trying to acquire a resource.")
        req = super().request()
        yield req
        logger.debug(f"Time {self._env.now:.2f}: Task '{task}' acquired a resource. Starting execution for {delay} cycles.")
        ins.record.exe_start_time.append(self._env.now)
        if self.checkneed():
            if attributes is None:
                self.data.append((task, self._env.now, len(self.queue), "req", "B"))
            else:
                self.data.append((task, self._env.now, len(self.queue), "req", "B", attributes))
        if v is None:
            yield self._env.timeout(delay)
        else:
            yield self._env.timeout(delay, value=v)
        if self.checkneed():
            if attributes is None:
                self.data.append((task, self._env.now, len(self.queue), "req", "E"))
            else:
                self.data.append((task, self._env.now, len(self.queue), "req", "E", attributes))
        ins.record.exe_end_time.append(self._env.now)
        super().release(req)
        logger.debug(f"Time {self._env.now:.2f}: Task '{task}' released a resource.")

    def execute(self, task, delay, ins, v=None, attributes=None):
        return self._env.process(self.exe(task, delay, ins, v, attributes))
        
ind2ins = []
def init_graph(program):
    global ind2ins
    for pos, pe in enumerate(program):
        index2ins = {}
        for id,inst in enumerate(pe):
            if inst.inst_type == 2 or inst.inst_type == 3:
                index2ins[inst.index] = inst
            # 只有一个前驱||recv,直接计入到waitinglast
            if inst.inst_type == 1 or inst.inst_type == 2 or inst.inst_type == 3 or inst.para_num+inst.feat_num == 1:
                inst.waitinglast = True
        ind2ins += [index2ins]
    for pe in program:
        for inst in pe:
            # send
            if inst.inst_typ == 2:
                assert inst.trigger_index == []
                position = inst.position
                index = inst.index
                insrecv = ind2ins[position][index]
                inst.next.append(insrecv)

# TODO
class Timer:
    # timersample.txt
    def __init__(self, env, time, cores):
        self.env = env
        self.time = time
        self.depth = 10 #最多回溯10条
        self.cores = cores
        self.env.process(self.run())
        # waitready设置为全局变量，因为一直在变
    def handler(self):
        # here waitready is a list of tasks waiting the last ready trigger condition in current block
        # we need backtrace the PSG
        for core in self.cores:
            for event in core.running_event:
                inst = core.event2task[event].inst
            # 目前我认为不用维护一个waitlist,这样的waitlist需要时常维护(尤其是remove),甚至可能不如直接每次遍历
            # 先把深度设置小一些,看看时间
                hot = self.dfs(inst, self.depth)-1
                if inst.inst_type == 1:
                    assert len(inst.next) == 0
                    assert inst.hot == 0
                    assert hot == 0
                inst.hot += hot
            for inst in core.running_send:
                assert inst.inst_type == 2
                print(inst.next)
                print(inst.next[0].next)
                hot = self.dfs(inst, self.depth)
                inst.hot += hot
    # 每次采样时触发,给最近的true,进行hot+1
    # 如果一个指令本身在waitinglast,那么此时
    # 能用sample尽量sample,实在不行，不用sample
    def dfs(self, inst, depth):
        if depth == 0:
            return 1
        nr = 1
        for inst_next in inst.next:
            if not inst_next.waitinglast:
                continue
            nr += self.dfs(inst_next, depth-1)
        return nr


    def run(self):
        # TODO:finish信号
        last_quotient = -1
        while True:
            yield self.env.timeout(self.time)
            current_quotient = self.env.now // 44542148
            if current_quotient != last_quotient:
                print("time:", self.env.now)
                last_quotient = current_quotient
            self.handler()
            self.finish = True
            for core in self.cores:
                # print(f"finish:{core.scheduler.finish}")
                self.finish = self.finish and core.scheduler.finish    
            if self.finish:
                print(self.finish)
                break

total_power = 0
power_summary = {
    "compute": 0,
    "dram_read": 0,
    "dram_write": 0,
    "noc_hop": 0,
    "sram_read": 0,
    "sram_write": 0,
    "cim_local_read": 0
}
power_trace_file = None
tpu_flop_power = 0
vect_flop_power = 0
dram_read_power = 0
dram_write_power = 0
noc_hop_power = 0
sram_write_power = 0
sram_read_power = 0
cim_local_read_power = 0

def init_power_trace(power_config_path, power_trace_path):
    global power_trace_file, tpu_flop_power, vect_flop_power, dram_read_power, dram_write_power, noc_hop_power, sram_write_power, sram_read_power, cim_local_read_power
    with open(power_config_path, 'r') as f:
        power_config = json.load(f)
        tpu_flop_power = power_config['tpu_flop_power']
        vect_flop_power = power_config['vect_flop_power']
        dram_read_power = power_config['dram_read_power']
        dram_write_power = power_config['dram_write_power']
        noc_hop_power = power_config['noc_hop_power']
        sram_write_power = power_config['sram_write_power']
        sram_read_power = power_config['sram_read_power']
        cim_local_read_power = power_config['cim_local_read_power']
    
    power_trace_file = open(power_trace_path, 'w')

def record_power_trace(time, device_id, task_id, power, event_type, feat_precision="", para_precision="", flops=""):
    global total_power
    total_power += power

    if "tpu_compute" in event_type:
        power_summary["compute"] += power
    elif "dram_read" in event_type:
        power_summary["dram_read"] += power
    elif "dram_write" in event_type:
        power_summary["dram_write"] += power
    elif "noc" in event_type:
        power_summary["noc_hop"] += power
    elif "sram_read" in event_type:
        power_summary["sram_read"] += power
    elif "sram_write" in event_type:
        power_summary["sram_write"] += power
    elif "cim_local_read" in event_type:
        power_summary["cim_local_read"] += power

    if power_trace_file:
        id_key = "dram_id" if "dram" in event_type else "core_id"
        log_entry = f"[time]:{time}, [{id_key}]:{device_id}, [task_id]:{task_id}, [power]:{power}, [type]:{event_type}"
        if feat_precision != "":
            log_entry += f", [feat_precision]:{feat_precision}"
        if para_precision != "":
            log_entry += f", [para_precision]:{para_precision}"
        if flops != "":
            log_entry += f", [flops]:{flops}"
        power_trace_file.write(log_entry + "\n")

def close_power_trace():
    if power_trace_file:
        power_trace_file.write(f"\n[total_power]:{total_power}\n\n")
        power_trace_file.write("Power Summary:\n")
        for key, value in power_summary.items():
            power_trace_file.write(f"  [{key}]: {value}\n")
        power_trace_file.close()
