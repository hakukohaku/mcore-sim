import os
import sys
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import json
from typing import List
from src import ArchConfig
from pydantic import BaseModel, ValidationError
from src.sim_type import DimSlice, Slice, Instruction, PEworkload, Workload, TaskType, DataType

def config_analyzer(filename: str) -> ArchConfig:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            config = ArchConfig.model_validate(data)
            return config
        except ValidationError as e:
            print(e.json())
            
arch_configs = config_analyzer(os.path.join(project_root, "arch", "gemini4_4.json"))
arch_configs.core.spm.size /= 4

class Core(BaseModel):
    x: int
    y: int

class Block(BaseModel):
    cores: List[Core]
    tensor_slice: List[DimSlice]

class IFeature(BaseModel):
    source: str
    source_layer_id: int
    blocks: List[Block]

class WFeature(BaseModel):
    source: str
    blocks: List[Block]

class OFeature(BaseModel):
    dest: str
    next: List[int]
    blocks: List[Block]

class Partition(BaseModel):
    dims: List[int]
    def num(self) -> int:
        res = None
        for dim_part in self.dims:
            if res == None:
                res = dim_part
            else:
                res = res * dim_part
        return res

class Layer(BaseModel):
    type: str
    layer_id: int
    layer_group_id: int
    group_num: int
    layer_batch_size: int
    output_partition: Partition
    # input_fetch实际上以输出为视角fetch，C维度的fetch不影响输入
    input_fetch: Partition
    input_feature: List[IFeature]
    wgt_feature: List[WFeature]
    output_feature: List[OFeature]

class Network(BaseModel):
    name: str
    batch_size: int
    layers: List[Layer]
    
def get_list_id(x: int, y: int) -> int:
    return x * arch_configs.core.y + y

if __name__ == "__main__":
    # ========== 数据精度配置 ==========
    # 设置feature和parameter的数据精度（以bytes为单位）
    # 可以根据实际需求修改这些值：
    # - 1: INT8 或自定义 8-bit 格式
    # - 2: FP16 或 BF16
    # - 4: FP32
    # - 8: FP64
    FEAT_PRECISION = 1  # Feature tensor精度，默认INT8 (1 byte)
    PARA_PRECISION = 1  # Parameter tensor精度，默认INT8 (1 byte)
    
    print(f"数据精度配置: Feature={FEAT_PRECISION} bytes, Parameter={PARA_PRECISION} bytes")
    # ==========================================
    
    #net = json_analyzer("tools/GEMM.json")
    core_num = arch_configs.core.x * arch_configs.core.y

    inf = 100000
    global_inst_id = 0
    pewls = [PEworkload(id=id) for id in range(core_num)]
    
    send_map = {}
    write_map = {}
    
    total_N = 64
    total_K = 2048
    total_M = 1024
    
    tile_num_n = 1
    tile_num_k = 8
    tile_num_m = 8
    
    tile_size_n = 64
    tile_size_k = 512
    tile_size_m = 256
    
    # 将矩阵k维度映射到空间阵列的y维度（行），m维度映射到x维度（列）
    temporal_loop_k = math.ceil(tile_num_k / arch_configs.core.y)
    temporal_loop_m = math.ceil(tile_num_m / arch_configs.core.x)
    
    n = 0
    cur_start_n = n * tile_size_n
    cur_size_n = min(tile_size_n, total_N - cur_start_n)
    # 遍历阵列，为每个核创建instruction
    for core_y in range(arch_configs.core.y):
        # 只有每行两边的mem_core直接访问DRAM的核，各读入一半的activation并通过path-based multicast广播给行内其他核
        # 当前tile所需的activation范围
        cur_start_k = ((temporal_loop_k-1)*arch_configs.core.y + core_y) * tile_size_k
        cur_size_k = min(tile_size_k, total_K - cur_start_k)
        act_range_left = Slice(tensor_slice=[
                DimSlice(start=cur_start_n, end=cur_start_n + cur_size_n), 
                DimSlice(start=cur_start_k, end=cur_start_k + cur_size_k/2), 
                DimSlice(start=0, end=1), 
                DimSlice(start=0, end=1)
            ])
        act_range_right = Slice(tensor_slice=[
                DimSlice(start=cur_start_n, end=cur_start_n+cur_size_n), 
                DimSlice(start=cur_start_k + cur_size_k/2, end=cur_start_k + cur_size_k), 
                DimSlice(start=0, end=1), 
                DimSlice(start=0, end=1)
            ])
        
        # 每行最左侧的核读取一半activation，并通过SEND最远端在行内进行广播
        list_core_id = get_list_id(0, core_y)
        global_inst_id += 1
        pewls[list_core_id].insts.append(
            Instruction(
                inst_type = TaskType.READ,
                index = global_inst_id,
                layer_id = 1,
                data_type = DataType.FEAT,
                tensor_slice = act_range_left.tensor_slice,
                feat_num = 0,
                feat_precision = FEAT_PRECISION,
                para_precision = PARA_PRECISION
            )
        )
        global_inst_id += 1
        pewls[list_core_id].insts.append(
            Instruction(
                inst_type = TaskType.SEND,
                index = global_inst_id,
                layer_id = 1,
                data_type = DataType.FEAT,
                position = get_list_id(arch_configs.core.x-1, core_y),
                tensor_slice = act_range_left.tensor_slice,
                feat_num = 1,
                feat_precision = FEAT_PRECISION,
                para_precision = PARA_PRECISION
            )
        )
        # 读指令之后trigger 广播的send指令
        pewls[list_core_id].insts[-2].trigger_index.append(global_inst_id)
        
        for broadcast_index in range(1,arch_configs.core.x):
            info = (list_core_id, get_list_id(broadcast_index, core_y), 1, "feat", act_range_left.tensor_slice[0].start, act_range_left.tensor_slice[1].start,act_range_left.tensor_slice[2].start,act_range_left.tensor_slice[3].start)
            send_map[info] = global_inst_id
            pewls[list_core_id].insts[-1].path_dst.append(get_list_id(broadcast_index, core_y))
        
        # 每行最右侧的核读取一半activation，并通过SEND最远端在行内进行广播
        list_core_id = get_list_id(arch_configs.core.x-1, core_y)
        global_inst_id += 1
        pewls[list_core_id].insts.append(
            Instruction(
                inst_type = TaskType.READ,
                index = global_inst_id,
                layer_id = 1,
                data_type = DataType.FEAT,
                tensor_slice = act_range_right.tensor_slice,
                feat_num = 0,
                feat_precision = FEAT_PRECISION,
                para_precision = PARA_PRECISION
            )
        )
        global_inst_id += 1
        pewls[list_core_id].insts.append(
            Instruction(
                inst_type = TaskType.SEND,
                index = global_inst_id,
                layer_id = 1,
                data_type = DataType.FEAT,
                position = get_list_id(0, core_y),
                tensor_slice = act_range_right.tensor_slice,
                feat_num = 1,
                feat_precision = FEAT_PRECISION,
                para_precision = PARA_PRECISION
            )
        )
        # 读指令之后trigger 广播的send指令
        pewls[list_core_id].insts[-2].trigger_index.append(global_inst_id)
        
        for broadcast_index in range(0,arch_configs.core.x-1):
            info = (list_core_id, get_list_id(broadcast_index, core_y), 1, "feat", act_range_right.tensor_slice[0].start, act_range_right.tensor_slice[1].start,act_range_right.tensor_slice[2].start,act_range_right.tensor_slice[3].start)
            send_map[info] = global_inst_id
            pewls[list_core_id].insts[-1].path_dst.append(get_list_id(broadcast_index, core_y))
            
        # 中间不能访问DRAM的核，等待数据广播
        for core_x in range(0, arch_configs.core.x):
            list_core_id = get_list_id(core_x, core_y)
            if core_x != 0: # 不是每行最左边的核，需要RECV被广播的act_left
                info = (get_list_id(0, core_y), list_core_id, 1, "feat", act_range_left.tensor_slice[0].start, act_range_left.tensor_slice[1].start,act_range_left.tensor_slice[2].start,act_range_left.tensor_slice[3].start) 
                pewls[list_core_id].insts.append(
                        Instruction(
                            inst_type = TaskType.RECV,
                            index = send_map[info],
                            layer_id = 1,
                            data_type = DataType.FEAT,
                            tensor_slice = act_range_left.tensor_slice,
                            feat_num = 1,
                            feat_precision = FEAT_PRECISION,
                            para_precision = PARA_PRECISION
                        )
                    )
            if core_x != arch_configs.core.x-1:
                info = (get_list_id(arch_configs.core.x-1, core_y), list_core_id, 1, "feat", act_range_right.tensor_slice[0].start, act_range_right.tensor_slice[1].start,act_range_right.tensor_slice[2].start,act_range_right.tensor_slice[3].start) 
                pewls[list_core_id].insts.append(
                        Instruction(
                            inst_type = TaskType.RECV,
                            index = send_map[info],
                            layer_id = 1,
                            data_type = DataType.FEAT,
                            tensor_slice = act_range_right.tensor_slice,
                            feat_num = 1,
                            feat_precision = FEAT_PRECISION,
                            para_precision = PARA_PRECISION
                        )
                    )
                
# 构建指令trigger关系
input_inst = [TaskType.READ, TaskType.RECV]
comp_inst = [TaskType.CONV, TaskType.POOL, TaskType.ELEM, TaskType.FC, TaskType.GCONV, TaskType.PTP, TaskType.TRANS]
output_inst = [TaskType.WRITE, TaskType.SEND]

wl = Workload(name="path_based_broadcast", pes=pewls)
workload_json = wl.model_dump_json(indent=4)

output_file_path = os.path.join(project_root, "tests", "communication", "path_based_broadcast.json")
output_dir = os.path.dirname(output_file_path)
os.makedirs(output_dir, exist_ok=True)

with open(output_file_path, "w") as file:
    print(workload_json, file=file)

print("路径广播指令生成完成！")
#print(f"所有指令已使用配置的数据精度: Feature={FEAT_PRECISION} bytes, Parameter={PARA_PRECISION} bytes")

