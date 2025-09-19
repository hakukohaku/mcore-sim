from enum import IntEnum, Enum
from typing import List, Optional, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

def ceil(a: int, b: int):
    return (a + b - 1) // b

class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class RouterFail(BaseModel):
    start_time: int
    end_time: int
    router_id: int
    times: int

class LinkFail(BaseModel):
    start_time: int
    end_time: int
    router_id: int
    direction: Direction
    times: int

class LsuFail(BaseModel):
    start_time: int
    end_time: int
    pe_id: int
    times: int

class TpuFail(BaseModel):
    start_time: int
    end_time: int
    pe_id: int
    times: int

class FailSlow(BaseModel):
    router: List[RouterFail]
    link: List[LinkFail]
    lsu: List[LsuFail]
    tpu: List[TpuFail]

class MsgType(IntEnum):
    DATA = 0
    MEM_REQUEST = 1

class TaskType(IntEnum):
    READ = 0
    WRITE = 1
    SEND = 2
    RECV = 3
    STAY = 4

    CONV = 5
    POOL = 6
    FC = 7
    ELEM = 8
    GCONV = 9
    PTP = 10
    TRANS = 11
    LOAD = 12
    STORE = 13
    FREE = 14

compute_task = [TaskType.CONV, TaskType.POOL, TaskType.FC, TaskType.ELEM, TaskType.GCONV, TaskType.PTP, TaskType.TRANS]

class AggrType(str, Enum):
    CONCAT_DIM_1 = "concat_dim_1"  # Aggregate by concatenating on the 1st dimension (index 1)
    CONCAT_DIM_0 = "concat_dim_0"  # Aggregate by concatenating on the 0th dimension (index 0)

class OperationType(IntEnum):
    CONV = 0
    POOL = 1
    FC = 2
    ELEM = 3

class DataType(IntEnum):
    PARA = 0
    FEAT = 1

class DimSlice(BaseModel):
    start: int
    end: int

class Slice(BaseModel):
    tensor_slice: List[DimSlice]
    def size(self) -> int:
        res = None
        for dim_slice in self.tensor_slice:
            dim_len = max(0, dim_slice.end - dim_slice.start)
            if res == None:
                res = dim_len
            else:
                res = res * dim_len
        return res
    
    def max(self, other: "Slice") -> "Slice":
        res = []
        for i in range(len(self.tensor_slice)):
            res.append(
                DimSlice(
                    start = min(self.tensor_slice[i].start, other.tensor_slice[i].start),
                    end = max(self.tensor_slice[i].end, other.tensor_slice[i].end)
                )
            )
        return Slice(tensor_slice=res)
    
class Data(BaseModel):
    index: int = -1
    tensor_slice: List[DimSlice] = []

    def __lt__(self, other: "Data") -> bool:
        return self.index < other.index

class MemReqPayload(BaseModel):
    # 对于core向非直连的dram读写数据，需要调用noc先向目标core发送读写请求，对应这里的MemReqPayload
    req_op: str # "Read" or "Write"
    requester_id: int
    original_index: int
    original_slice: List[DimSlice]
    # 对于写操作，数据需要预先通过NoC发送，这里只发送元数据
    # size: int
    data_type: DataType
    layer_id: int

class Task(BaseModel):
    layer_id: int = -1
    opcode: str
    index: int
    tensor_slice: List[DimSlice]
    flops: int = 0
    num_operands: int
    feat_num: int = 0
    para_num: int = 0
    feat: List[Data] = []
    para: List[Data] = []
    inst: "Instruction" = None
    # 数据精度，以byte为单位，默认为1
    feat_precision: int = 1
    para_precision: int = 1
    
    def size(self) -> int:
        cur_slice = Slice(tensor_slice=self.tensor_slice)
        return cur_slice.size()

class Nop(Task):
    def run(self, core, ins):
        ins.record.exe_start_time.append(core.env.now)
        yield core.env.timeout(0, self.index)
        ins.record.exe_end_time.append(core.env.now)

class IOTask(Task):
    target_dram_id: int = -1
    num_operands: int = 0
    original_requester_id: int = -1

    def size(self):
        """
        根据Instruction的data_type判断tensor类型并乘以相应的数据精度
        """
        base_size = super().size()  # 获取基本的元素个数
        
        # 根据对应Instruction的data_type判断当前tensor是feature还是parameter
        if self.inst and self.inst.data_type == DataType.FEAT:
            # Feature tensor使用feature精度
            return base_size * self.feat_precision
        elif self.inst and self.inst.data_type == DataType.PARA:
            # Parameter tensor使用parameter精度
            return base_size * self.para_precision
        else:
            # 如果没有对应的Instruction或data_type未设置，使用默认precision
            # 这种情况下可能是旧版本的代码或测试代码
            return base_size * self.feat_precision

    def input_size(self):
        raise NotImplementedError(f"{self.opcode} 类未实现 input_size 方法")

    def output_size(self):
        raise NotImplementedError(f"{self.opcode} 类未实现 output_size 方法")

    def run(self, core, ins):
        ins.record.ready_run_time.append(core.env.now)
        ins.record.pe_id = core.id
        yield core.env.process(core.spm_manager.allocate(self.opcode+str(self.index), self.output_size()))
        yield core.lsu.execute(self.opcode+str(self.index),ceil(self.size(), core.lsu_bandwidth), ins, self.index)
        core.env.process(core.spm_manager.free(self.opcode+str(self.index), self.input_size()))

class MemTask(Task):
    """
    对外部存储（Dram）的读写、释放任务
    """
    target_dram_id: int = -1
    num_operands: int = 1
    original_requester_id: int = -1
    data_type: DataType = DataType.FEAT # 代理任务需要继承此属性

    def size_in_bytes(self) -> int:
        base_size = self.size()
        precision = self.feat_precision if self.data_type == DataType.FEAT else self.para_precision
        return base_size * precision

    def run(self, core, ins, dram):
        # 代理任务的专属执行路径
        if self.original_requester_id != -1: # 是代理任务，接受请求帮助别的core进行read。
            # 代理任务已经被路由到正确的内存核心，直接在此核心上执行
            # 假设每个内存核心只连接一个DRAM
            if not core.dram_list:
                raise ValueError(f"Proxy task running on Core {core.id}, but it has no connected DRAM.")
            
            target_dram_id = core.dram_list[0]
            logger.debug(f"Time {core.env.now:.2f}: Core {core.id} executing PROXY access to its local DRAM {target_dram_id}")

            ins.record.exe_start_time.append(core.env.now)
            if self.opcode.lower() == "read":
                yield core.env.process(core.spm_manager.allocate(self.opcode + str(self.index), self.output_size()))
                yield from dram[target_dram_id].read(self.size_in_bytes(), ins)
            elif self.opcode.lower() == "write":
                yield core.env.process(core.spm_manager.free(self.opcode + str(self.index), self.input_size()))
                yield from dram[target_dram_id].write(self.size_in_bytes(), ins)
            ins.record.exe_end_time.append(core.env.now)
            return # 代理任务执行完毕，结束

        """
        执行内存任务的流程：
        1. 请求DRAM资源。
        2. 根据mem_opcode执行相应的DRAM操作（read/write/free）。
        3. 释放DRAM资源（由resource内部的.execute函数自动处理）。
        """
        # 记录任务准备信息
        ins.record.ready_run_time.append(core.env.now)
        ins.record.pe_id = core.id

        # 1. 确定目标DRAM和拥有它的核心
        target_mem_core_id = -1
        target_dram_id = -1

        if self.target_dram_id != -1:
            # 如果指令指定了DRAM ID，则查找拥有它的核心
            target_dram_id = self.target_dram_id
            for c in core.arch.cores:
                if target_dram_id in c.dram_list:
                    target_mem_core_id = c.id
                    break
            if target_mem_core_id == -1:
                raise ValueError(f"Instruction specified DRAM {target_dram_id}, but no core is connected to it.")
        else:
            # 如果指令未指定，则找最近的
            target_mem_core_id, target_dram_id = core.get_dram_index()

        ins.record.dram_id = target_dram_id

        # 2. 判断是本地执行还是远程请求
        is_local = (target_mem_core_id == core.id)

        if is_local:
            # --- 本地执行 ---
            logger.debug(f"Time {core.env.now:.2f}: Core {core.id} performing LOCAL access to DRAM {target_dram_id}")
            
            ins.record.exe_start_time.append(core.env.now)
            
            if self.opcode.lower() == "read":
                yield core.env.process(core.spm_manager.allocate(self.opcode + str(self.index), self.output_size()))
                yield from dram[target_dram_id].read(self.size_in_bytes(), ins)
            
            elif self.opcode.lower() == "write":
                yield core.env.process(core.spm_manager.free(self.opcode + str(self.index), self.input_size()))
                yield from dram[target_dram_id].write(self.size_in_bytes(), ins)

            ins.record.exe_end_time.append(core.env.now)
        else:
            # --- 远程请求 ---
            print(f"Time {core.env.now:.2f}: Core {core.id} initiating REMOTE access to DRAM {target_dram_id} via Core {target_mem_core_id}")
            logger.debug(f"Time {core.env.now:.2f}: Core {core.id} initiating REMOTE access to DRAM {target_dram_id} via Core {target_mem_core_id}")
            
            # 1. 创建并发送MemoryRequestMessage
            payload = MemReqPayload(
                req_op=self.opcode,
                requester_id=core.id,
                original_index=self.index,
                original_slice=self.tensor_slice,
                data_type=ins.data_type,
                layer_id=ins.layer_id
            )
            
            mem_request_msg = Message(
                src=core.id,
                dst=target_mem_core_id,
                msg_type=MsgType.MEM_REQUEST,
                payload=payload,
                ins=ins,  # 将当前指令添加到消息中
                data=Data(index=self.index, tensor_slice=[DimSlice(start=0,end=1)]) # 添加最小化data
            )
            print(f"Time {core.env.now:.2f}: Core {core.id} initiating REMOTE access to DRAM {target_dram_id} via Core {target_mem_core_id}")
            
            ins.record.exe_start_time.append(core.env.now)
            yield core.data_out.put(mem_request_msg)

            if self.opcode.lower() == "read":
                # 2. 创建并注册一个衍生的RECV任务来等待数据返回
                derivative_recv_task = DerivativeRecv(
                    layer_id=self.layer_id,
                    opcode="Recv",
                    index=self.index,
                    tensor_slice=self.tensor_slice,
                    inst=self.inst,
                    done_event=core.env.event()
                )
                core.pending_recvs[self.index] = derivative_recv_task
                
                # 3. 等待衍生的RECV任务完成（即其done_event被触发）
                print(f"Time {core.env.now:.2f}: Core {core.id} is now PAUSED, waiting for data (index: {self.index}) from Core {target_mem_core_id}.")
                yield derivative_recv_task.done_event
                
                # 4. 数据已返回，任务完成
                print(f"Time {core.env.now:.2f}: Core {core.id} has RESUMED, data (index: {self.index}) received.")
                ins.record.exe_end_time.append(core.env.now)
            
class ComputeTask(Task):
    layer_id: int
    num_operands: int = 2
    feat_aggr_type: Optional[AggrType] = None

    def input_size(self):
        res = 0
        for input in self.feat:
            input_slice = Slice(tensor_slice=input.tensor_slice)
            res += input_slice.size()
            
        for wgt in self.para:
            wgt_slice = Slice(tensor_slice=wgt.tensor_slice)
            res += wgt_slice.size()
        return res + self.size()

    def output_size(self):
        return self.size()
    
    def calc_flops(self):
        raise NotImplementedError(f"{self.opcode} 类未实现 calc_flops 方法")
    
    def run(self, core, ins):
        from src.common import tpu_flop_power, record_power_trace, sram_read_power, cim_local_read_power
        self.calc_flops()
        ins.record.ready_run_time.append(core.env.now)
        ins.record.pe_id = core.id

        # SRAM read power for features
        feat_size = sum(Slice(tensor_slice=d.tensor_slice).size() for d in self.feat) * self.feat_precision
        if feat_size > 0:
            power = feat_size * sram_read_power
            record_power_trace(core.env.now, core.id, self.index, power, "sram_read_feat")

        # SRAM read power for parameters (CIM local read)
        para_size = sum(Slice(tensor_slice=d.tensor_slice).size() for d in self.para) * self.para_precision
        if para_size > 0:
            power = para_size * cim_local_read_power
            record_power_trace(core.env.now, core.id, self.index, power, "cim_local_read_para")

        # log_prefix = f"Time {core.env.now:.2f}: Core{core.id} [Task {self.index}]"
        # logger.debug(f"{log_prefix} - ComputeTask.run START")

        #为output准备空间
        yield core.env.process(core.spm_manager.allocate(self.opcode+str(self.index), self.output_size()))
        
        # logger.debug(f"Time {core.env.now:.2f}: Core{core.id} [Task {self.index}] - ALLOC complete, starting TPU execute.")

        #执行tpu计算
        yield core.tpu.execute(self.opcode+str(self.index), ceil(self.flops, core.tpu_flops), ins, self.index)

        power = self.flops * tpu_flop_power
        record_power_trace(core.env.now, core.id, self.index, power, "tpu_compute", self.feat_precision, self.para_precision, self.flops)

        # logger.debug(f"Time {core.env.now:.2f}: Core{core.id} [Task {self.index}] - TPU EXEC complete.")

        #释放input空间
        core.env.process(core.spm_manager.free(self.opcode+str(self.index), self.input_size()))
        # logger.debug(f"Time {core.env.now:.2f}: Core{core.id} [Task {self.index}] - ComputeTask.run END (free initiated).")
    
class Record(BaseModel):
    exe_start_time: List[int] = []
    exe_end_time: List[int] = []
    ready_run_time: List[int] = []
    # 记录多个指令的唤醒
    mulins: List[int] = []
    # 记录指令执行的PE
    pe_id: int = -1
    dram_id: int = -1

class CommunicationTask(Task):
    dst: int
    src: int
    num_operands: int = 0

class Instruction(BaseModel):
    inst_type: TaskType
    index: int
    trigger_index: List[int] = []
    # 只有WRITE指令会用到
    trigger_core_id: List[int] = []
    layer_id: int
    group_num: int = 1
    data_type: DataType
    position: int = 0         # SEND的终点，或RECV的源
    path_dst: List[int] = []  # 路径广播目标列表，默认为空列表
    tensor_slice: List[DimSlice]
    feat_num: int = 0
    para_num: int = 0
    feat_aggr_type: Optional[AggrType] = None
    # 数据精度，以byte为单位，默认为1
    feat_precision: int = 1
    para_precision: int = 1
    target_dram_id: int = -1

    # 在想应该累计每个block对后面造成的影响，这样的热点或许更有效
    start_time: int = -1
    record: Record = Record()
    # 目前我没想细化这些，ready到finish都是running
    # 通过last_trigger_tree反向搜索，找到第一个running的指令,并将它作为性能的瓶颈
    # ready: bool = False
    running: bool = False
    # 在pre_analysis中将真的只有1个来wait的置为1
    waitinglast: bool = False
    # finish: bool = False
    hot: int = 0
    next: List["Instruction"] = []
    # 以及每个指令造成的影响是一样的吗？
    # tensor_slice is unused in hash
    def trig(self):
        self.ready = True

    def run(self):
        self.running = True

    def addhot(self, hot):
        self.hot += hot

    def __eq__(self, other):
        if not isinstance(other, Instruction):
            return NotImplemented
        return (self.inst_type, self.index, self.layer_id, self.data_type, self.position) == \
               (other.inst_type, other.index, other.layer_id, other.data_type, other.position)

    def __hash__(self):
        return hash((self.inst_type, self.index, self.layer_id, self.data_type, self.position))
    
class Message(BaseModel):
    """
    消息类，支持路径广播功能
    
    使用示例：
    # 普通消息（只发送给目标核心）
    msg = Message(ins=instruction, src=0, data=data, dst=5)
    
    # 路径广播消息（发送给目标核心和路径上的其他核心）
    msg = Message(ins=instruction, src=0, data=data, dst=5, path_dst=[1, 2, 3])
    # 这将把消息发送给核心5(dst),同时也会传递给核心1、2、3
    """
    ins: Optional[Instruction] = None
    src: int
    data: Optional[Data] = None
    dst: int
    path_dst: List[int] = []  # 路径广播目标列表，默认为空列表
    msg_type: MsgType = MsgType.DATA
    payload: Optional[MemReqPayload] = None
    route_strategy: str = "wormhole"

    def __lt__(self, other: "Message") -> bool:
        if self.data is not None and other.data is not None:
            return self.data < other.data
        # 提供一个回退的比较方法，避免在data为None时出错
        return self.src < other.src
    
    def should_deliver_to_core(self, core_id: int) -> bool:
        """
        判断消息是否应该传递给指定的核心
        如果dst或path_dst中包含core_id, 则返回True
        """
        return core_id == self.dst or core_id in self.path_dst

class Operation(BaseModel):
    operation: str
    layer_id: int

class PEworkload(BaseModel):
    id: int
    insts: List[Instruction] = []

class Workload(BaseModel):
    name: str
    pes: List[PEworkload] = []

class Read(MemTask):
    opcode: str = "Read"

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()

class Write(MemTask):
    opcode: str = "Write"

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Conv(ComputeTask):
    opcode: str = "Conv"
    def calc_flops(self):
        wgt_slice = Slice(tensor_slice=self.para[0].tensor_slice)
        wgt_H = wgt_slice.tensor_slice[2].end - wgt_slice.tensor_slice[2].start
        wgt_W = wgt_slice.tensor_slice[3].end - wgt_slice.tensor_slice[3].start

        self.flops = self.size() * wgt_H * wgt_W

class Pool(ComputeTask):
    opcode: str = "Pool"
    def calc_flops(self):
        self.flops = self.size() * 4
    
class Elem(ComputeTask):
    opcode: str = "Elem"
    def calc_flops(self):
        self.flops = self.size()

class FC(ComputeTask):
    opcode: str = "FC"
    def calc_flops(self):
        """
        Calculates FLOPs for a fully-connected layer (matrix multiplication).
        Supports aggregating multiple input features (activations) based on `feat_aggr_type`.
        FLOPs = N * M * K
        - N is the batch/sequence dimension from input features.
        - M is the output feature dimension from the weight tensor.
        - K is the input feature dimension from input features/weight tensor.
        """
        if not self.para:
            raise ValueError(f"FC.calc_flops() for task {self.index} requires parameter input (self.para), but it is empty.")
        if not self.feat:
            raise ValueError(f"FC.calc_flops() for task {self.index} requires feature input (self.feat), but it is empty.")

        # According to my_geninst_GEMM.py, the weight tensor (parameter) has a shape of (M, K).
        # The feature tensor (activation) has a shape of (N, K).
        # The output tensor has a shape of (N, M).
        
        if len(self.para[0].tensor_slice) < 2:
            raise ValueError(f"FC task {self.index}: parameter tensor must be at least 2D, but got {len(self.para[0].tensor_slice)} dimensions.")
        
        # Get M from the weight tensor's first dimension (index 0).
        m_size = self.para[0].tensor_slice[0].end - self.para[0].tensor_slice[0].start
        
        # Get K from the weight tensor's second dimension (index 1).
        k_from_para = self.para[0].tensor_slice[1].end - self.para[0].tensor_slice[1].start

        if self.feat_aggr_type is None:
            # Default case: No aggregation. Expects a single feature input.
            if len(self.feat) != 1:
                raise ValueError(f"FC task {self.index} with no feat_aggr_type expects 1 feature input, but got {len(self.feat)}.")
            
            feat_slice = self.feat[0].tensor_slice
            if len(feat_slice) < 2:
                raise ValueError(f"FC task {self.index}: feature tensor must be at least 2D, but got {len(feat_slice)} dimensions.")

            n_size = feat_slice[0].end - feat_slice[0].start
            k_from_feat = feat_slice[1].end - feat_slice[1].start
            
            if k_from_feat != k_from_para:
                raise ValueError(f"FC task {self.index}: K dimension mismatch between feature ({k_from_feat}) and parameter ({k_from_para}).")
            
            self.flops = n_size * m_size * k_from_para

        elif self.feat_aggr_type == AggrType.CONCAT_DIM_0:
            # Concatenate features along dimension 0 (N). Dimension 1 (K) must be consistent across all features.
            n_total = 0
            k_first = -1
            for i, feat_data in enumerate(self.feat):
                feat_slice = feat_data.tensor_slice
                if len(feat_slice) < 2:
                    raise ValueError(f"FC task {self.index} CONCAT_DIM_0: feature input {i} must be at least 2D.")
                
                n_total += feat_slice[0].end - feat_slice[0].start
                k_current = feat_slice[1].end - feat_slice[1].start
                
                if k_first == -1:
                    k_first = k_current
                elif k_first != k_current:
                    raise ValueError(f"FC task {self.index} with CONCAT_DIM_0: K dimensions of all features must be equal, but got varying sizes.")
            
            if k_first != k_from_para:
                raise ValueError(f"FC task {self.index}: Aggregated feature K dimension ({k_first}) mismatches parameter K dimension ({k_from_para}).")

            self.flops = n_total * m_size * k_from_para

        elif self.feat_aggr_type == AggrType.CONCAT_DIM_1:
            # Concatenate features along dimension 1 (K). Dimension 0 (N) must be consistent across all features.
            k_total = 0
            n_first = -1
            for i, feat_data in enumerate(self.feat):
                feat_slice = feat_data.tensor_slice
                if len(feat_slice) < 2:
                    raise ValueError(f"FC task {self.index} CONCAT_DIM_1: feature input {i} must be at least 2D.")
                
                k_total += feat_slice[1].end - feat_slice[1].start
                n_current = feat_slice[0].end - feat_slice[0].start

                if n_first == -1:
                    n_first = n_current
                elif n_first != n_current:
                    raise ValueError(f"FC task {self.index} with CONCAT_DIM_1: N dimensions of all features must be equal, but got varying sizes.")
            
            if k_total != k_from_para:
                raise ValueError(f"FC task {self.index}: Aggregated feature K dimension ({k_total}) mismatches parameter K dimension ({k_from_para}).")

            self.flops = n_first * m_size * k_from_para
        else:
            raise NotImplementedError(f"Unsupported feat_aggr_type: {self.feat_aggr_type} for FC task {self.index}")

class GConv(ComputeTask):
    opcode: str = "GConv"
    group_num: int
    def calc_flops(self):
        wgt_slice = Slice(tensor_slice=self.para[0].tensor_slice)
        wgt_H = wgt_slice.tensor_slice[2].end - wgt_slice.tensor_slice[2].start
        wgt_W = wgt_slice.tensor_slice[3].end - wgt_slice.tensor_slice[3].start

        self.flops = self.size() * wgt_H * wgt_W
        self.flops //= self.group_num

class PTP(ComputeTask):
    opcode: str = "PTP"
    def calc_flops(self):
        self.flops = self.size() * 7

class Trans(ComputeTask):
    opcode: str = "Trans"
    def calc_flops(self):
        self.flops = 0

class Stay(Task):
    opcode: str = "Stay"
    flops: int = -1
    def run(self, core):
        yield core.env.timeout(0)

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Send(CommunicationTask):
    opcode: str = "Send"
    src: int = -1
    path_dst: List[int] = []  # 路径广播目标列表，默认为空列表
    
    def run(self, core, ins):
        from src.common import record_power_trace, sram_read_power
        logger.debug(f"Time {core.env.now:.2f}: Core {core.id} [Send.run] START for index {self.index}")
        # SRAM read power for sending data
        size = self.size() * (self.feat_precision if ins.data_type == DataType.FEAT else self.para_precision)
        if size > 0:
            power = size * sram_read_power
            record_power_trace(core.env.now, core.id, self.index, power, "sram_read_send")

        # 分析时send/recv合并处理，因为index一致
        # 记录
        ins.record.pe_id = core.id
        ins.record.ready_run_time.append(core.env.now)
        # 为Output准备空间
        yield core.env.process(core.spm_manager.allocate(self.opcode+str(self.index), self.output_size()))
        ins.record.exe_start_time.append(core.env.now)
        # 将 Message 对象放入 Core 的数据输出通道 (data_out)，包含路径广播信息
        logger.debug(f"Time {core.env.now:.2f}: Core {core.id} [Send.run] About to PUT message for index {self.index} into data_out link.")
        yield core.data_out.put(Message(data=Data(index=self.index, tensor_slice=self.tensor_slice), dst=self.dst, src=core.id, ins=ins, path_dst=self.path_dst))
        logger.debug(f"Time {core.env.now:.2f}: Core {core.id} [Send.run] FINISHED PUT for index {self.index}")

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Recv(CommunicationTask):
    opcode: str = "Recv"
    dst: int = -1
    src: int = -1
    def run(self, core, ins):
        ins.record.pe_id = core.id
        ins.record.exe_end_time.append(core.env.now)

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()
    
class DerivativeTask:
    """一个空的Mixin类，用于标记所有在运行时动态生成的任务。"""
    pass

class DerivativeSend(Send, DerivativeTask):
    """一个在运行时生成的衍生SEND任务，例如，用于在远程读取后将数据发回。"""
    proxy_ins: Instruction

class DerivativeRecv(Recv, DerivativeTask):
    """一个在运行时生成的衍生RECV任务，用于等待远程读操作的返回数据。"""
    done_event: Any # This will hold a simpy.Event

    class Config:
        arbitrary_types_allowed = True

class Store(Task):
    opcode: str = "Store"
    num_operands: int = 0

    def run(self, core, ins):
        ins.record.exe_start_time.append(core.env.now)
        # Allocate space for the tensor in SPM. After this, the data is considered available.
        yield core.env.process(core.spm_manager.allocate(self.opcode + str(self.index), self.size()))
        ins.record.exe_end_time.append(core.env.now)

    def input_size(self):
        return 0

    def output_size(self):
        return self.size()

class Load(Task):
    opcode: str = "Load"
    num_operands: int = 0

    def run(self, core, ins):
        # This is a logical operation. It assumes data is already in SPM.
        # Its completion triggers subsequent tasks. It consumes no time.
        ins.record.exe_start_time.append(core.env.now)
        yield core.env.timeout(0)
        ins.record.exe_end_time.append(core.env.now)

    def input_size(self):
        return 0

    def output_size(self):
        return 0

class Free(Task):
    opcode: str = "Free"
    num_operands: int = 0

    def run(self, core, ins):
        ins.record.exe_start_time.append(core.env.now)
        # Explicitly free a tensor slice from SPM.
        yield core.env.process(core.spm_manager.free(self.opcode + str(self.index), self.size()))
        ins.record.exe_end_time.append(core.env.now)
    
    def input_size(self):
        return self.size()

    def output_size(self):
        return 0

