import simpy
import os
import sys
import logging
from functools import partial, wraps
from src.core import Core
from src.noc_new import NoC, Link, Direction
from src.arch_config import CoreConfig, NoCConfig, ArchConfig, LinkConfig, MemConfig
from src.dram import Dram
from src.sim_type import *
from src import common
from src.draw import draw_grid
from analysis.trace_format import *
from typing import List
logger = logging.getLogger("Arch")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def trace(env, callback):
    """Replace the ``step()`` method of *env* with a tracing function
    that calls *callbacks* with an events time, priority, ID and its
    instance just before it is processed.

    """
    def get_wrapper(env_step, callback):
        """Generate the wrapper for env.step()."""
        @wraps(env_step)
        def tracing_step():
            """Call *callback* for the next event if one exist before
            calling ``env.step()``."""
            if len(env._queue):
                t, prio, eid, event = env._queue[0]
                callback(t, prio, eid, event)
            return env_step()
        return tracing_step

    env.step = get_wrapper(env.step, callback)

data = []

def monitor(data, t, prio, eid, event):
    if (isinstance(event, simpy.resources.store.StoreGet) or isinstance(event, simpy.resources.store.StorePut)) and False:
        logger.info((t, eid, event.proc._generator, type(event), event.resource, event.value))
    # else:
        # logger.info((t, eid, type(event), event.value))
    data.append((t, eid, type(event)))

monitor = partial(monitor, data)

def patch_resource(resource, pre=None, post=None):
    """Patch *resource* so that it calls the callable *pre* before each
    put/get/request/release operation and the callable *post* after each
    operation.  The only argument to these functions is the resource
    instance.

    """
    def get_wrapper(func):
        # Generate a wrapper for put/get/request/release
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is the actual wrapper
            # Call "pre" callback
            if pre:
                pre(resource, func)

            # Perform actual operation
            ret = func(*args, **kwargs)

            # Call "post" callback
            if post:
                post(resource, ret)

            return ret
        return wrapper

    # Replace the original operations with our wrapper
    for name in ['put', 'get', 'request', 'release']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))

def monitor1(data, resource, func):
    """This is our monitoring callback."""
    item = (
        resource._env.now,  # The current simulation time
        "pre",
        len(resource.items),  # The number of queued processes
        resource.items,
        # resource,
        func,
    )
    # logger.info(item)
    data.append(item)

def monitor2(data, resource, ret):
    """This is our monitoring callback."""
    item = (
        resource._env.now,  # The current simulation time
        "post",
        len(resource.items),  # The number of queued processes
        resource.items,
        # resource,
        ret,
    )
    # logger.info(item)
    data.append(item)

monitor1 = partial(monitor1, data)
monitor2 = partial(monitor2, data)

class Arch:
    def __init__(self, arch: ArchConfig, program: List[List[Instruction]], fail: FailSlow, net_name: str, fail_kind: str, stage=None):
        print("Constructing hardware architecture.")
        self.env = simpy.Environment()
        self.stage = stage
        
        common.init_power_trace('power/power_config/power.json', 'power/power_trace/power_trace.txt')
        
        self.mem_type = arch.mem.type
        self.noc = self.build_noc(arch.noc)
        self.dram = self.build_dram(arch.mem)
        
        #print(len(self.noc.r2r_links))
        if stage == "pre_analysis":
            common.init_graph(program)
        self.program = program

        self.cores = self.build_cores(arch.core, program, arch.mem)
        self.core_x = arch.core.x
        self.core_y = arch.core.y
        
        self.layer_start = [-1 for _ in range(25)]
        self.layer_end = [-1 for _ in range(25)]

        self.net_name = net_name
        self.end_time = 0
        self.fail_kind = fail_kind
        
        trace(self.env, monitor)
        patch_resource(self.cores[0].data_in.store, pre=monitor1, post=monitor2)

        self.fail_slow = fail
        print("Construction finished.")

    def debug(self):
        for d in data:
            logger.info(d)

    def link_fail(self, fail: LinkFail):
        yield self.env.timeout(fail.start_time)
        match fail.direction:
            case Direction.NORTH:
                self.noc.routers[fail.router_id].north_in.change_delay(fail.times)
                self.noc.routers[fail.router_id].north_out.change_delay(fail.times)
            case Direction.SOUTH:
                self.noc.routers[fail.router_id].south_in.change_delay(fail.times)
                self.noc.routers[fail.router_id].south_out.change_delay(fail.times)
            case Direction.EAST:
                self.noc.routers[fail.router_id].east_in.change_delay(fail.times)
                self.noc.routers[fail.router_id].east_out.change_delay(fail.times)
            case Direction.WEST:
                self.noc.routers[fail.router_id].west_in.change_delay(fail.times)
                self.noc.routers[fail.router_id].west_out.change_delay(fail.times)
        
        yield self.env.timeout(fail.end_time-fail.start_time)
        match fail.direction:
            case Direction.NORTH:
                self.noc.routers[fail.router_id].north_in.recover_delay(fail.times)
                self.noc.routers[fail.router_id].north_out.recover_delay(fail.times)
            case Direction.SOUTH:
                self.noc.routers[fail.router_id].south_in.recover_delay(fail.times)
                self.noc.routers[fail.router_id].south_out.recover_delay(fail.times)
            case Direction.EAST:
                self.noc.routers[fail.router_id].east_in.recover_delay(fail.times)
                self.noc.routers[fail.router_id].east_out.recover_delay(fail.times)
            case Direction.WEST:
                self.noc.routers[fail.router_id].west_in.recover_delay(fail.times)
                self.noc.routers[fail.router_id].west_out.recover_delay(fail.times)

    def router_fail(self, fail: RouterFail):
        yield self.env.timeout(fail.start_time)
        self.noc.routers[fail.router_id].router_fail(fail.times)
        yield self.env.timeout(fail.end_time-fail.start_time)
        self.noc.routers[fail.router_id].router_recover(fail.times)

    def lsu_fail(self, fail: LsuFail):
        yield self.env.timeout(fail.start_time)
        self.cores[fail.pe_id].lsu_fail(fail.times)
        yield self.env.timeout(fail.end_time-fail.start_time)
        self.cores[fail.pe_id].lsu_recover(fail.times)

    def tpu_fail(self, fail: TpuFail):
        yield self.env.timeout(fail.start_time)
        self.cores[fail.pe_id].tpu_fail(fail.times)
        yield self.env.timeout(fail.end_time-fail.start_time)
        self.cores[fail.pe_id].tpu_recover(fail.times)

    def run_fail_slow(self):
        for link_fail in self.fail_slow.link:
            self.env.process(self.link_fail(link_fail))
        
        for router_fail in self.fail_slow.router:
            self.env.process(self.router_fail(router_fail))

        for lsu_fail in self.fail_slow.lsu:
            self.env.process(self.lsu_fail(lsu_fail))

        for tpu_fail in self.fail_slow.tpu:
            self.env.process(self.tpu_fail(tpu_fail))

    def build_cores(self, config: CoreConfig, program: List[List[Instruction]], mem_config: MemConfig) -> List[Core]:
        cores = []
        for id in range(config.x * config.y):
            link1 = Link(self.env, LinkConfig(width=8192, delay=1))
            link2 = Link(self.env, LinkConfig(width=8192, delay=1))
            core = Core(self.env, config, program[id], id, self, link1, link2, self.stage)

            self.noc.routers[id].bound_with_core(link1, link2)
            cores.append(core)
            # TODO:timer should be in second stage
        if self.stage == "post_analysis":
            self.timer = common.Timer(self.env, 20000, cores)

        for id in range(config.x * config.y):
            cores[id].scheduler.bound_cores(cores)

        if mem_config.type == "all_core_distributed":
            for id in range(len(cores)):
                # 每个核心都连接到一个DRAM，DRAM的ID与核心ID相同
                cores[id].dram_list.append(id)
        elif mem_config.type == "two_sides_edge":
            # 只有特定的核心连接到DRAM
            if not hasattr(mem_config, 'mem_core'):
                 raise ValueError("DRAM type is 'two_sides_edge' but 'mem_core' is not defined in config.")
            for core_id_str, dram_info in mem_config.mem_core.items():
                core_id = int(core_id_str)
                if core_id < len(cores):
                    cores[core_id].dram_list.append(dram_info['dram_id'])

        return cores

    def build_noc(self, config: NoCConfig) -> NoC:
        print("Building NoC architecture.")
        return NoC(self.env, config).build_connection()
    
    def build_dram(self, config: MemConfig) -> List[Dram]:
        print("Building Dram")
        dram = []
        if config.type == "all_core_distributed":
            for id in range(config.num):
                dram.append(Dram(self.env, config, id))
        elif config.type == "two_sides_edge":
            if not hasattr(config, 'mem_core') or not config.mem_core:
                raise ValueError("DRAM type is 'two_sides_edge' but 'mem_core' is not defined or is empty in config.")
            
            # 找出最大的dram_id来确定列表大小
            all_dram_ids = [d['dram_id'] for d in config.mem_core.values()]
            max_dram_id = max(all_dram_ids)
            
            # 创建一个正确大小的列表，用None填充
            dram = [None] * (max_dram_id + 1)
            
            # 将DRAM实例填充到正确的位置
            for core_id_str, dram_info in config.mem_core.items():
                dram_id = dram_info['dram_id']
                dram[dram_id] = Dram(self.env, config, dram_id)
        else:
            raise ValueError(f"Unsupported DRAM distribution type: {config.type}")
            
        return dram

    # 输出可视化文件
    def make_print_lsu():
        # 对于每个lsu
        count = 0
        # req->count+=1 release->count-=1
    def processesmonitor(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            for idx, line in enumerate(data):
                task,ts, lenthqueue, ation, ph = line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\": \"{task}\",\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"lenthqueue\":{lenthqueue}}}}}")

    # 这个由学长来编号,对于每个编号(id)怎么处理的逻辑我已经写好了:
    def processesmonitorlink(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            for idx, line in enumerate(data):
                task,ts, lenthqueue, ation, ph,dest = line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\": \"{task}\",\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"lenthqueue\":{lenthqueue},\"dest\":{dest}}}}}")

    def processesspm(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            for idx, line in enumerate(data):
                task, capacity, action, size ,ts , ph= line
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\":\"{task}\" ,\"ph\":\"{ph}\",\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"act\":\"{action}\",\"capacity\":{capacity},\"size\":{size}}}}}")

    def processesflow(self,data,file,id,source):
        if len(data)==0:
            return
        with open(file,"w") as f: 
            for idx, line in enumerate(data):
                index, _, action, ts= line
                task=action+str(index)
                # 如果不是第一行，则在行前添加逗号和换行符
                if idx != 0:
                    f.write(",\n")
                f.write(f"{{\"name\":\"{task}\" ,\"ph\":\"B\",\"ts\":{ts},\"id\":{index},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"act\":\"{action}\"}}}}")
                f.write(",\n")
                f.write(f"{{\"name\":\"{task}\" ,\"ph\":\"E\",\"ts\":{ts},\"id\":{index},\"pid\":{id},\"tid\":\"{source}\",\"args\":{{\"act\":\"{action}\"}}}}")
                if action!="recv":
                    f.write(",\n")
                    f.write(f"{{\"name\":\"connect\" ,\"ph\":\"s\",\"ts\":{ts},\"id\":{index},\"pid\":{id},\"tid\":\"{source}\"}}")
                else:
                    f.write(",\n")
                    f.write(f"{{\"name\":\"connect\",\"ph\":\"f\",\"bp\":\"e\",\"id\":{index},\"ts\":{ts},\"pid\":{id},\"tid\":\"{source}\"}}")


                
                            
    def make_print(self):
        #print(self.cores[0].lsu.data)
        #print(self.cores[1].tpu.data)
        #print(self.cores[2].lsu.data)
        #print(self.noc.routers[0].core_out.linkentry.data)
        #print(self.cores[1].spm_manager.data)
        os.makedirs("gen", exist_ok=True)
        for i in range(len(self.cores)):
            self.processesmonitor(self.cores[i].lsu.data,"gen/lsu"+str(i)+".json",i,"lsu")
            self.processesmonitor(self.cores[i].tpu.data,"gen/tpu"+str(i)+".json",i,"tpu")
            self.processesspm(self.cores[i].spm_manager.data,"gen/spm"+str(i)+".json",i,"spm")

        for i in range(len(self.noc.r2r_links)):
            self.processesmonitorlink(self.noc.r2r_links[i].linkentry.data,"gen/link"+str(i)+".json",i,"link")

        
        if common.cfg.flow:
            for i in range(len(self.cores)):
                self.processesflow(self.cores[i].flow_in,"gen/flow_in"+str(i)+".json",i,"flow_in")
            for i in range(len(self.cores)):
                self.processesflow(self.cores[i].flow_out,"gen/flow_out"+str(i)+".json",i,"flow_out")

    def draw(self):
        L, H = 4, 4
        data = [
            [3, 1, 1, 2],
            [3, 3, 2, 3],
            [1, 1, 3, 3],
            [3, 3, 3, 1]
        ]
        links = []
        for i in self.noc.r2r_links:
            if i.tag == 1:
                links.append((i.corefrom, i.coreto, i.hop))
            else:
                print(i.tag)
        draw_grid(L, H, data, links)

    # 输出采集的数据（two stages）
    def process(self):
        for pos, pe in enumerate(self.program):
            for id, inst in enumerate(pe):
                print(pos, inst.index, inst.inst_type, inst.hot)

    def output_data(self, net: str, fail: str):
        print("Writing performance data...")

        fail = fail.split("/")
        fail = fail[-1].split(".")[0]

        file_path = os.path.join("data/", net)
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        file_path = os.path.join(file_path, fail)
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        inst_file = os.path.join(file_path, "inst_info.txt")
        compute_trace = []
        
        # inst_index -> record 
        comm_record = {}
        # inst_index -> core_id (用于跟踪SEND指令的core)
        send_core_map = {}
        comm_trace = []
        
        # 添加调试信息
        total_insts = sum(len(insts) for insts in self.program)
        print(f"Debug: Total instructions to process: {total_insts}")

        with open(inst_file, "w") as file:
            print(f"Debug: Starting to process {total_insts} instructions", file=file)
            
            # 阶段1：预处理 - 收集所有SEND指令
            print("Phase 1: Collecting all SEND instructions...")
            print(f"Debug: Phase 1 - Collecting all SEND instructions", file=file)
            for core_id, insts in enumerate(self.program):
                for inst in insts:
                    if inst.inst_type == TaskType.SEND:
                        # 记录SEND指令的Core ID，方便后续调试
                        send_core_map[inst.index] = core_id
                        comm_record[inst.index] = inst.record
            
            print(f"Phase 1 complete: Collected {len(comm_record)} SEND instructions")
            print(f"Debug: Phase 1 complete - Collected {len(comm_record)} SEND instructions", file=file)
            
            # 阶段2：正式处理 - 处理所有指令
            print("Phase 2: Processing all instructions...")
            print(f"Debug: Phase 2 - Processing all instructions", file=file)
            for core_id, insts in enumerate(self.program):
                for inst in insts:
                    # 没有被执行的指令
                    if inst.record.exe_start_time == []:
                        error_msg = f"Error: Instruction{inst.index} (Core{core_id}, Type:{inst.inst_type}) not executed."
                        print(f"Error: Instruction{inst.index} (Core{core_id}, Type:{inst.inst_type}) not executed.", file=file)
                    
                    if inst.inst_type in compute_task:
                        # 添加安全检查
                        if not inst.record.ready_run_time:
                            error_msg = f"Error: Compute instruction {inst.index} (Core{core_id}, Type:{inst.inst_type}) has empty ready_run_time"
                            print(error_msg, file=file)
                            print(error_msg)  # 同时输出到终端
                            continue
                        if not inst.record.exe_end_time:
                            error_msg = f"Error: Compute instruction {inst.index} (Core{core_id}, Type:{inst.inst_type}) has empty exe_end_time"
                            print(error_msg, file=file)
                            print(error_msg)  # 同时输出到终端
                            continue
                        if not inst.record.exe_start_time:
                            error_msg = f"Error: Compute instruction {inst.index} (Core{core_id}, Type:{inst.inst_type}) has empty exe_start_time"
                            print(error_msg, file=file)
                            print(error_msg)  # 同时输出到终端
                            continue
                        
                        assert len(inst.record.ready_run_time) > 0
                        assert len(inst.record.exe_end_time) == 1
                        assert len(inst.record.exe_start_time) == 1
                        print(f"Instruction{inst.index}: type {inst.inst_type}, layer_id {inst.layer_id}, pe_id {inst.record.pe_id}", file=file)
                        print(f"    ready_time {inst.record.ready_run_time[0]}, exe_time {inst.record.exe_end_time[0]-inst.record.exe_start_time[0]}, end_time {inst.record.exe_start_time[0]}", file=file)
                        print(f"    operands_time: {inst.record.mulins}", file=file)
                        
                        compute_trace.append(
                            InstTrace(
                                instruction_id = inst.index,
                                instruction_type = inst.inst_type,
                                layer_id = inst.layer_id,
                                pe_id = inst.record.pe_id,
                                start_time = inst.record.exe_start_time[0],
                                end_time = inst.record.exe_end_time[0]
                            )
                        )
                    else:
                        if inst.inst_type == TaskType.SEND:
                            # SEND指令已经在阶段1处理过了，这里跳过
                            continue
                        elif inst.inst_type == TaskType.RECV:
                            if inst.index in comm_record:
                                # 检查SEND指令的exe_start_time是否为空
                                if not comm_record[inst.index].exe_start_time:
                                    send_core_id = send_core_map.get(inst.index, 'Unknown')
                                    error_msg = f"Error: SEND instruction {inst.index} (Core{send_core_id}, Type:{TaskType.SEND}) has empty exe_start_time list"
                                    detail_msg1 = f"  SEND record: pe_id={comm_record[inst.index].pe_id}"
                                    detail_msg2 = f"  RECV record: Core{core_id}, pe_id={inst.record.pe_id}, exe_start_time={inst.record.exe_start_time}, exe_end_time={inst.record.exe_end_time}"
                                    
                                    print(error_msg, file=file)
                                    print(detail_msg1, file=file)
                                    print(detail_msg2, file=file)
                                    
                                    print(error_msg)  # 同时输出到终端
                                    print(detail_msg1)
                                    print(detail_msg2)
                                    continue
                                
                                # 检查RECV指令的exe_end_time是否为空
                                if not inst.record.exe_end_time:
                                    error_msg = f"Error: RECV instruction {inst.index} (Core{core_id}, Type:{inst.inst_type}) has empty exe_end_time list"
                                    detail_msg1 = f"  SEND record: pe_id={comm_record[inst.index].pe_id}, exe_start_time={comm_record[inst.index].exe_start_time}"
                                    detail_msg2 = f"  RECV record: pe_id={inst.record.pe_id}"
                                    
                                    print(error_msg, file=file)
                                    print(detail_msg1, file=file)
                                    print(detail_msg2, file=file)
                                    
                                    print(error_msg)  # 同时输出到终端
                                    print(detail_msg1)
                                    print(detail_msg2)
                                    continue
                                
                                comm_trace.append(
                                    CommInst(
                                        instruction_id = inst.index,
                                        instruction_type = inst.inst_type,
                                        # 当前层的id
                                        layer_id = inst.layer_id,
                                        pe_id = inst.record.pe_id,
                                        start_time = comm_record[inst.index].exe_start_time[0],
                                        end_time = inst.record.exe_end_time[0],
                                        src_id = comm_record[inst.index].pe_id,
                                        dst_id = inst.record.pe_id
                                    )
                                )
                            else: # 找不到相应的SEND指令
                                print(f"Warning: No corresponding SEND instruction found for RECV instruction {inst.index} (Core{core_id}, Type:{inst.inst_type})", file=file)
                                # 对于没有对应SEND的RECV指令，使用自身的时间信息
                                if inst.record.exe_start_time and inst.record.exe_end_time:
                                    comm_trace.append(
                                        CommInst(
                                            instruction_id = inst.index,
                                            instruction_type = inst.inst_type,
                                            layer_id = inst.layer_id,
                                            pe_id = inst.record.pe_id,
                                            start_time = inst.record.exe_start_time[0],
                                            end_time = inst.record.exe_end_time[0],
                                            src_id = -1,  # 未知源
                                            dst_id = inst.record.pe_id
                                        )
                                    )
                                else:
                                    print(f"Warning: RECV instruction {inst.index} (Core{core_id}, Type:{inst.inst_type}) has empty time records", file=file)
                                    print(f"  exe_start_time: {inst.record.exe_start_time}", file=file)
                                    print(f"  exe_end_time: {inst.record.exe_end_time}", file=file)
                        # READ/WRITE
                        else:
                            if not inst.record.exe_start_time or not inst.record.exe_end_time:
                                error_msg = f"Error: {inst.inst_type} instruction {inst.index} (Core{core_id}, Type:{inst.inst_type}) has empty time records"
                                detail_msg1 = f"  exe_start_time: {inst.record.exe_start_time}"
                                detail_msg2 = f"  exe_end_time: {inst.record.exe_end_time}"
                                detail_msg3 = f"  pe_id: {inst.record.pe_id}"
                                
                                print(error_msg, file=file)
                                print(detail_msg1, file=file)
                                print(detail_msg2, file=file)
                                print(detail_msg3, file=file)
                                
                                print(error_msg)  # 同时输出到终端
                                print(detail_msg1)
                                print(detail_msg2)
                                print(detail_msg3)
                                continue
                            
                            comm_trace.append(
                                CommInst(
                                    instruction_id = inst.index,
                                    instruction_type = inst.inst_type,
                                    # 当前层的id
                                    layer_id = inst.layer_id,
                                    pe_id = inst.record.pe_id,
                                    start_time = inst.record.exe_start_time[0],
                                    end_time = inst.record.exe_end_time[0]
                                )
                            )

        compute_trace = CompTrace(trace=compute_trace)
        comp_json_file = os.path.join(file_path, "comp_trace.json")
        with open(comp_json_file, "w") as file:
            comp_json = compute_trace.model_dump_json(indent=4)
            print(comp_json, file=file)

        comm_trace = CommTrace(trace=comm_trace)
        comp_json_file = os.path.join(file_path, "comm_trace.json")
        with open(comp_json_file, "w") as file:
            comp_json = comm_trace.model_dump_json(indent=4)
            print(comp_json, file=file)

        layer_file = os.path.join(file_path, "layer_info.txt") 
        with open(layer_file, "w") as file:
            for id, time in enumerate(self.layer_start):
                print(f"Layer{id} start at {time}.", file=file)
                print(f"Layer{id} end at {self.layer_end[id]}.", file=file)

        print("Finished.")

    def run(self):
        print("Start simulation.")
        
        self.run_fail_slow()

        self.env.run()

        for id in range(len(self.cores)):
            self.end_time = max(self.end_time, self.cores[id].end_time)
            print(f"Core{id} end time: {self.cores[id].end_time}")
            print(f"Core{id} processed [{self.cores[id].scheduler.inst_counter}/{len(self.cores[id].program)}] instructions.")
            print(f"Max buffer usage is {self.cores[id].spm_manager.max_buf}. [{self.cores[id].spm_manager.container.capacity-self.cores[id].spm_manager.container.level}/{self.cores[id].spm_manager.container.capacity}]")

        print("Simulation finished.")
        # 将值传入json文件
        # self.make_print()
        if self.stage == "post_analysis":
            self.process()

        self.output_data(self.net_name, self.fail_kind)

        print(f"Total power: {common.total_power}")
        print("Power Summary:")
        for key, value in common.power_summary.items():
            print(f"  [{key}]: {value}")
        common.close_power_trace()
        
        # self.draw()

        return self.env