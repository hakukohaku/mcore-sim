import os
import sys
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import json
from typing import List
from src import ArchConfig
from pydantic import BaseModel, ValidationError
from src.sim_type import DimSlice, Slice, Instruction, PEworkload, Workload, TaskType, DataType, AggrType

def config_analyzer(filename: str) -> ArchConfig:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            config = ArchConfig.model_validate(data)
            return config
        except ValidationError as e:
            print(e.json())

arch_configs = config_analyzer("../arch/myarch_gemini1_4_cim.json")
arch_configs.core.spm.size /= 4
core_num = arch_configs.core.x * arch_configs.core.y

inf = 100000
global_inst_id = 0
pewls = [PEworkload(id=id) for id in range(core_num)]
    
send_map = {}
write_map = {}

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

def json_analyzer(filename: str) -> Network:
    with open(filename, 'r') as file:
        data = json.load(file)
        try:
            net = Network.model_validate(data)
            return net
        except ValidationError as e:
            print(e.json())

def get_list_id(core_x, core_y):
    return core_y * arch_configs.core.x + core_x

def get_core_coor(core_id):
    core_x = core_id % arch_configs.core.x
    core_y = core_id // arch_configs.core.x
    return core_x, core_y

def intersect(a: Slice, b: Slice) -> Slice:
    new_slice = []
    for id in range(len(a.tensor_slice)):
        new_slice.append(
            DimSlice(
                start = max(a.tensor_slice[id].start, b.tensor_slice[id].start), 
                end = min(a.tensor_slice[id].end, b.tensor_slice[id].end)
            )
        )
    res = Slice(tensor_slice=new_slice)
    return res

def time_range(a: Slice, lb_size: int, time: int) -> Slice:
    batch_start = lb_size * time + a.tensor_slice[0].start
    batch_end = lb_size * time + a.tensor_slice[0].end
    a.tensor_slice[0] = DimSlice(start=batch_start, end=batch_end)
    return a

# 对输出而言，四个维度都可能随fetch改变
def output_fetch_range(a: Slice, fetch_sch: Partition, step: int) -> Slice:
    if fetch_sch.num() == 1:
        return a
    
    dim_sizes, dim_lens, dim_poses = [], [], []

    for (i, dim_slice) in enumerate(a.tensor_slice):
        dim_size = 1
        for j in range(i+1, len(fetch_sch.dims)):
            dim_size = dim_size * fetch_sch.dims[j]
        dim_sizes.append(dim_size)
        dim_lens.append((dim_slice.end-dim_slice.start)//fetch_sch.dims[i])

        if i == 0:
            # print(f"{step}%{fetch_sch.num()} // {dim_sizes[i]}")
            dim_poses.append((step%fetch_sch.num())//dim_sizes[i])
        else:
            # print(f"{step}%{dim_sizes[i-1]} // {dim_sizes[i]}")
            dim_poses.append((step%dim_sizes[i-1])//dim_sizes[i])
        
        # print(f"{i} {dim_poses[i]} {dim_sizes[i]} {dim_lens[i]}")

    new_slice = []
    for id in range(len(a.tensor_slice)):
        remain = (a.tensor_slice[id].end - a.tensor_slice[id].start) % dim_lens[id]
        if remain == 0:
            start = a.tensor_slice[id].start + dim_poses[id] * dim_lens[id]
            end = a.tensor_slice[id].start + min(dim_poses[id]*dim_lens[id]+dim_lens[id], a.tensor_slice[id].end)
            new_slice.append(DimSlice(start=start, end=end))
        else:
            start = a.tensor_slice[id].start + dim_poses[id] * dim_lens[id] + (dim_poses[id] if dim_poses[id] < remain else remain)
            end = a.tensor_slice[id].start + min(dim_poses[id]*dim_lens[id]+dim_lens[id]+(dim_poses[id]+1 if dim_poses[id]+1<remain else remain), a.tensor_slice[id].end)
            new_slice.append(DimSlice(start=start, end=end))

    return Slice(tensor_slice=new_slice)

# 对输入而言，C1维度的大小不随fetch的C改变
def input_fetch_range(a: Slice, fetch_sch: Partition, step: int) -> Slice:
    if fetch_sch.num() == 1:
        return a
    
    dim_sizes, dim_lens, dim_poses = [], [], []

    for (i, dim_slice) in enumerate(a.tensor_slice):
        dim_size = 1
        for j in range(i+1, len(fetch_sch.dims)):
            dim_size = dim_size * fetch_sch.dims[j]
        dim_sizes.append(dim_size)
        dim_lens.append((dim_slice.end-dim_slice.start)//fetch_sch.dims[i])

        if i == 0:
            dim_poses.append((step%fetch_sch.num())//dim_sizes[i])
        else:
            dim_poses.append((step%dim_sizes[i-1])//dim_sizes[i])
    
    new_slice = []
    for id in range(len(a.tensor_slice)):
        if id != 1:
            remain = (a.tensor_slice[id].end - a.tensor_slice[id].start) % dim_lens[id]
            if remain == 0:
                start = a.tensor_slice[id].start + dim_poses[id] * dim_lens[id]
                end = a.tensor_slice[id].start + min(dim_poses[id]*dim_lens[id]+dim_lens[id], a.tensor_slice[id].end)
                new_slice.append(DimSlice(start=start, end=end))
            else:
                start = a.tensor_slice[id].start + dim_poses[id] * dim_lens[id] + (dim_poses[id] if dim_poses[id] < remain else remain)
                end = a.tensor_slice[id].start + min(dim_poses[id]*dim_lens[id]+dim_lens[id]+(dim_poses[id]+1 if dim_poses[id]+1<remain else remain), a.tensor_slice[id].end)
                new_slice.append(DimSlice(start=start, end=end))
        else:
            # C维度不变
            new_slice.append(a.tensor_slice[id])

    return Slice(tensor_slice=new_slice)

# 对权重而言，只有fetch的C维度有影响（影响第二维）
def wgt_fetch_range(a: Slice, fetch_sch: Partition, step: int) -> Slice:
    if fetch_sch.dims[1] == 1:
        return a
    
    # fetch后C维度的大小
    len_c = (a.tensor_slice[1].end - a.tensor_slice[1].start) // fetch_sch.dims[1]
    # 当前step对应的是C维度的第几块
    size_c = fetch_sch.num() // fetch_sch.dims[0]
    pos_c = (step % size_c) // (fetch_sch.dims[2] * fetch_sch.dims[3])
    # print(f"step:{step}, size_c:{size_c}, pos_c:{pos_c}")
    # 计算该块的C维度范围（最后一块需要均分给前面的块）
    remain = (a.tensor_slice[1].end - a.tensor_slice[1].start) % len_c
    if remain == 0:
        start = a.tensor_slice[1].start + pos_c * len_c
        end = a.tensor_slice[1].start + min(pos_c*len_c+len_c, a.tensor_slice[1].end)
        a.tensor_slice[1] = DimSlice(start=start, end=end)
    else:
        start = a.tensor_slice[1].start + pos_c * len_c + (pos_c if pos_c < remain else remain)
        end = a.tensor_slice[1].start + min(pos_c*len_c+len_c+(pos_c+1 if pos_c+1<remain else remain), a.tensor_slice[1].end)
        a.tensor_slice[1] = DimSlice(start=start, end=end)

    return a

def get_input_size(layer: Layer):
    layer_input_part = layer.input_feature[0].blocks[0].tensor_slice
    layer_input_part = Slice(tensor_slice=layer_input_part)
    # print("input-before")
    # print(layer_input_part)
    # print(layer.input_fetch)
    layer_input_part = input_fetch_range(layer_input_part, layer.input_fetch, 0)
    # print("input-after")
    # print(layer_input_part)
    return layer_input_part.size()

def get_spm_size(layer: Layer):
    spm_size = 0
    input_size = get_input_size(layer)
    spm_size += input_size
    
    if layer.wgt_feature[0].blocks:
        layer_wgt_part = layer.wgt_feature[0].blocks[0].tensor_slice
        layer_wgt_part = Slice(tensor_slice=layer_wgt_part)
        # print("wgt-before")
        # print(layer_wgt_part)
        layer_wgt_part = wgt_fetch_range(layer_wgt_part, layer.input_fetch, 0)
        # print("wgt-agter")
        # print(layer_wgt_part)
        spm_size += layer_wgt_part.size()

    # 计算当前分块下的output_slice
    layer_output_part = layer.output_feature[0].blocks[0].tensor_slice
    layer_output_part = Slice(tensor_slice=layer_output_part)
    layer_output_part = output_fetch_range(layer_output_part, layer.input_fetch, 0)
    spm_size += layer_output_part.size()

    return spm_size

def GEMM(read_activation=True, read_weight=True, write_back=True):
    global global_inst_id, pewls, send_map, write_map, lid, layer
    print(f"Generating inst of layer{lid}.")
    total_N = 48
    total_K = 48
    total_M = 192
    
    tile_num_n = 1
    tile_num_k = 1
    tile_num_m = 4
    
    tile_size_n = 64
    tile_size_k = 48
    tile_size_m = 48
    
    # weight stationary, 以weight tensor为主映射到空间core上。
    # 将矩阵k维度映射到空间阵列的y维度（行），m维度映射到x维度（列）
    temporal_loop_k = math.ceil(tile_num_k / arch_configs.core.y)
    temporal_loop_m = math.ceil(tile_num_m / arch_configs.core.x)
    
    send_map = {}
    
    for n in range(tile_num_n):
        cur_start_n = n * tile_size_n
        cur_size_n = min(tile_size_n, total_N - cur_start_n)
        
        # 追踪指令用于后续建立trigger关系
        act_providers_map = {}  # core_id -> tp_k -> [act_read_recv_insts]
        wgt_read_map = {}       # core_id -> (tp_k, tp_m) -> wgt_read_inst
        comp_inst_map = {}      # core_id -> (tp_k, tp_m) -> comp_inst
        elem_inst_map = {}      # core_id -> (tp_k, tp_m) -> elem_inst

        for tp_m in range(temporal_loop_m): # M-X维度时域循环
            for tp_k in range(temporal_loop_k): # K-Y维度时域循环
                if read_activation:
                    # 读入activation
                    # 硬件架构：只有边缘的核可以直接访问外部DRAM
                    if arch_configs.mem.type == "two_sides_edge": 
                        # 遍历阵列，为每个核创建instruction
                        for core_y in range(arch_configs.core.y):
                            # 只有每行两边的mem_core直接访问DRAM的核，各读入一半的activation并通过path-based multicast广播给行内其他核
                            # 当前tile所需的activation范围
                            cur_start_k = (tp_k*arch_configs.core.y + core_y) * tile_size_k
                            cur_size_k = min(tile_size_k, total_K - cur_start_k)
                            act_range_left = Slice(tensor_slice=[
                                    DimSlice(start=cur_start_n, end=cur_start_n + cur_size_n), 
                                    DimSlice(start=cur_start_k, end=cur_start_k + cur_size_k/2), 
                                    DimSlice(start=0, end=1), 
                                    DimSlice(start=0, end=1)
                                ])
                            act_range_right = Slice(tensor_slice=[
                                    DimSlice(start=cur_start_n, end=cur_start_n+cur_size_n), 
                                    DimSlice(start=cur_start_k + cur_size_k/2, end=cur_start_k + cur_size_k/2 + cur_size_k), 
                                    DimSlice(start=0, end=1), 
                                    DimSlice(start=0, end=1)
                                ])
                            
                            # 每行最左侧的核读取一半activation，并通过SEND最远端在行内进行广播
                            list_core_id = get_list_id(0, core_y)
                            
                            read_act_inst = Instruction(
                                inst_type = TaskType.READ,
                                index = global_inst_id,
                                layer_id = lid,
                                data_type = DataType.FEAT,
                                tensor_slice = act_range_left.tensor_slice
                            )
                            pewls[list_core_id].insts.append(read_act_inst)
                            if list_core_id not in act_providers_map: act_providers_map[list_core_id] = {}
                            if tp_k not in act_providers_map[list_core_id]: act_providers_map[list_core_id][tp_k] = []
                            act_providers_map[list_core_id][tp_k].append(read_act_inst)

                            global_inst_id += 1
                            send_inst = Instruction(
                                inst_type = TaskType.SEND,
                                index = global_inst_id,
                                layer_id = lid,
                                data_type = DataType.FEAT,
                                feat_num = 1,
                                position = get_list_id(arch_configs.core.x-1, core_y),
                                tensor_slice = act_range_left.tensor_slice
                            )
                            pewls[list_core_id].insts.append(send_inst)
                            # 读指令之后trigger 广播的send指令
                            pewls[list_core_id].insts[-2].trigger_index.append(global_inst_id)
                            
                            src_core = list_core_id
                            for broadcast_index in range(1,arch_configs.core.x):
                                dst_core = get_list_id(broadcast_index, core_y)
                                pewls[src_core].insts[-1].path_dst.append(dst_core)
                                
                                # 不是每行最左边的核，需要RECV被广播的act_left
                                recv_inst = Instruction(
                                    inst_type = TaskType.RECV,
                                    index = global_inst_id,
                                    layer_id = lid,
                                    data_type = DataType.FEAT,
                                    feat_num = 0,
                                    tensor_slice = act_range_left.tensor_slice
                                )
                                pewls[dst_core].insts.append(recv_inst)
                                if dst_core not in act_providers_map: act_providers_map[dst_core] = {}
                                if tp_k not in act_providers_map[dst_core]: act_providers_map[dst_core][tp_k] = []
                                act_providers_map[dst_core][tp_k].append(recv_inst)

                            global_inst_id += 1
        
                            
                            # 每行最右侧的核读取一半activation，并通过SEND最远端在行内进行广播
                            list_core_id = get_list_id(arch_configs.core.x-1, core_y)
                            
                            read_act_inst = Instruction(
                                inst_type = TaskType.READ,
                                index = global_inst_id,
                                layer_id = lid,
                                data_type = DataType.FEAT,
                                tensor_slice = act_range_right.tensor_slice
                            )
                            pewls[list_core_id].insts.append(read_act_inst)
                            if list_core_id not in act_providers_map: act_providers_map[list_core_id] = {}
                            if tp_k not in act_providers_map[list_core_id]: act_providers_map[list_core_id][tp_k] = []
                            act_providers_map[list_core_id][tp_k].append(read_act_inst)
                            
                            global_inst_id += 1
                            send_inst = Instruction(
                                inst_type = TaskType.SEND,
                                index = global_inst_id,
                                layer_id = lid,
                                data_type = DataType.FEAT,
                                feat_num = 1,
                                position = get_list_id(0, core_y),
                                tensor_slice = act_range_right.tensor_slice
                            )
                            pewls[list_core_id].insts.append(send_inst)
                            
                            # 读指令之后trigger 广播的send指令
                            pewls[list_core_id].insts[-2].trigger_index.append(global_inst_id)
                            
                            src_core = list_core_id
                            for broadcast_index in range(0,arch_configs.core.x-1):
                                dst_core = get_list_id(broadcast_index, core_y)
                                pewls[src_core].insts[-1].path_dst.append(dst_core)
                                # 不是每行最左边的核，需要RECV被广播的act_left
                                recv_inst = Instruction(
                                    inst_type = TaskType.RECV,
                                    index = global_inst_id,
                                    layer_id = lid,
                                    data_type = DataType.FEAT,
                                    feat_num = 0,
                                    tensor_slice = act_range_right.tensor_slice
                                )
                                pewls[dst_core].insts.append(recv_inst)
                                if dst_core not in act_providers_map: act_providers_map[dst_core] = {}
                                if tp_k not in act_providers_map[dst_core]: act_providers_map[dst_core][tp_k] = []
                                act_providers_map[dst_core][tp_k].append(recv_inst)

                            global_inst_id += 1
                            
                if read_weight:
                    for core_x in range(arch_configs.core.x):
                        for core_y in range(arch_configs.core.y):
                            list_core_id = get_list_id(core_x, core_y)
                            cur_start_k = (tp_k*arch_configs.core.y + core_y) * tile_size_k
                            cur_size_k = min(tile_size_k, total_K - cur_start_k)
                            cur_start_m = (tp_m*arch_configs.core.x + core_x) * tile_size_m
                            cur_size_m = min(tile_size_m, total_M - cur_start_m)
                            weight_range = Slice(tensor_slice=[
                                DimSlice(start=cur_start_m, end=cur_start_m + cur_size_m), 
                                DimSlice(start=cur_start_k, end=cur_start_k + cur_size_k), 
                                DimSlice(start=0, end=1), 
                                DimSlice(start=0, end=1)
                            ])
                            
                            read_wgt_inst = Instruction(
                                inst_type = TaskType.READ,
                                index = global_inst_id,
                                layer_id = lid,
                                data_type = DataType.PARA,
                                tensor_slice = weight_range.tensor_slice,
                                para_num = 0
                            )
                            pewls[list_core_id].insts.append(read_wgt_inst)
                            if list_core_id not in wgt_read_map: wgt_read_map[list_core_id] = {}
                            wgt_read_map[list_core_id][(tp_k, tp_m)] = read_wgt_inst
                            global_inst_id += 1

                # 添加计算指令
                for core_x in range(arch_configs.core.x):
                    for core_y in range(arch_configs.core.y):
                        list_core_id = get_list_id(core_x, core_y)
                        # Re-calculate dimensions for this specific core's tile
                        cur_start_k = (tp_k * arch_configs.core.y + core_y) * tile_size_k
                        cur_size_k = min(tile_size_k, total_K - cur_start_k)
                        cur_start_m = (tp_m * arch_configs.core.x + core_x) * tile_size_m
                        cur_size_m = min(tile_size_m, total_M - cur_start_m)
                        
                        # Output slice for the computation
                        output_range = Slice(tensor_slice=[
                            DimSlice(start=cur_start_n, end=cur_start_n + cur_size_n),
                            DimSlice(start=cur_start_m, end=cur_start_m + cur_size_m),
                            DimSlice(start=0, end=1),
                            DimSlice(start=0, end=1)
                        ])
                        
                        comp_inst = Instruction(
                            inst_type=TaskType.FC,
                            index=global_inst_id,
                            layer_id=lid,
                            data_type=DataType.FEAT,
                            feat_num=2 if read_activation else 0,
                            para_num=1, # read_weight=false时，不需要weight
                            tensor_slice=output_range.tensor_slice,
                            feat_aggr_type=AggrType.CONCAT_DIM_1
                        )
                        pewls[list_core_id].insts.append(comp_inst)
                        if list_core_id not in comp_inst_map: comp_inst_map[list_core_id] = {}
                        comp_inst_map[list_core_id][(tp_k, tp_m)] = comp_inst
                        global_inst_id += 1

                        # 只在 tp_k > 0时才需要累加，才生成ELEM指令
                        if tp_k > 0:
                            elem_inst = Instruction(
                                inst_type=TaskType.ELEM,
                                index=global_inst_id,
                                layer_id=lid,
                                data_type=DataType.FEAT,
                                feat_num=2, # 后续的累加总是需要2个输入
                                para_num=0,
                                tensor_slice=output_range.tensor_slice # 输入输出slice相同
                            )
                            pewls[list_core_id].insts.append(elem_inst)
                            if list_core_id not in elem_inst_map: elem_inst_map[list_core_id] = {}
                            elem_inst_map[list_core_id][(tp_k, tp_m)] = elem_inst
                            global_inst_id += 1

            # 在 tp_k 循环结束后，处理并链接列内规约 (Reduce)
            root_y = arch_configs.core.y // 2
            
            # 追踪规约阶段的指令
            reduce_send_map = {} # (x,y) -> send_inst
            reduce_recv_map = {} # (x,y) -> [recv_insts]
            reduce_elem_map = {} # (x,y) -> [elem_insts]
            write_back_inst_map = {} # x -> write_inst

            # 假设 x 维度上的所有列同时进行规约
            for core_x in range(arch_configs.core.x):
                
                # --- 为SEND/RECV对生成共享Index ---
                # 向上规约链 (core 0 -> 1 -> ... -> root)
                for sender_y in range(root_y):
                    receiver_y = sender_y + 1
                    
                    shared_index = global_inst_id
                    
                    # 1. 生成SEND指令
                    # --- FIX: Re-calculate output_range instead of using a non-existent attribute ---
                    cur_start_m = (tp_m * arch_configs.core.x + core_x) * tile_size_m
                    cur_size_m = min(tile_size_m, total_M - cur_start_m)
                    output_range = Slice(tensor_slice=[
                        DimSlice(start=cur_start_n, end=cur_start_n + cur_size_n),
                        DimSlice(start=cur_start_m, end=cur_start_m + cur_size_m),
                        DimSlice(start=0, end=1),
                        DimSlice(start=0, end=1)
                    ])
                    send_inst = Instruction(
                        inst_type=TaskType.SEND,
                        index=shared_index,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        feat_num=1,
                        position=get_list_id(core_x, receiver_y),
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[get_list_id(core_x, sender_y)].insts.append(send_inst)
                    reduce_send_map[(core_x, sender_y)] = send_inst
                    
                    # 2. 生成配对的RECV指令 (使用相同的index)
                    recv_inst = Instruction(
                        inst_type=TaskType.RECV,
                        index=shared_index,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        feat_num=0,
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[get_list_id(core_x, receiver_y)].insts.append(recv_inst)
                    if (core_x, receiver_y) not in reduce_recv_map: reduce_recv_map[(core_x, receiver_y)] = []
                    reduce_recv_map[(core_x, receiver_y)].append(recv_inst)

                    global_inst_id += 1 # 为SEND/RECV对增加一次ID

                    # 2.5 新增：为RECV的数据生成FREE指令，由RECV触发
                    free_inst = Instruction(
                        inst_type=TaskType.FREE,
                        index=global_inst_id,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[get_list_id(core_x, receiver_y)].insts.append(free_inst)
                    recv_inst.trigger_index.append(free_inst.index)
                    global_inst_id += 1

                    # 3. 为接收方生成ELEM指令
                    elem_inst = Instruction(
                        inst_type=TaskType.ELEM,
                        index=global_inst_id,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        feat_num=1, # 简化模型：只依赖自己的部分和
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[get_list_id(core_x, receiver_y)].insts.append(elem_inst)
                    if (core_x, receiver_y) not in reduce_elem_map: reduce_elem_map[(core_x, receiver_y)] = []
                    reduce_elem_map[(core_x, receiver_y)].append(elem_inst)
                    
                    global_inst_id += 1

                # 向下规约链 (core N-1 -> N-2 -> ... -> root)
                for sender_y in range(arch_configs.core.y - 1, root_y, -1):
                    receiver_y = sender_y - 1

                    shared_index = global_inst_id

                    # 1. 生成SEND指令
                    # --- FIX: Re-calculate output_range instead of using a non-existent attribute ---
                    cur_start_m = (tp_m * arch_configs.core.x + core_x) * tile_size_m
                    cur_size_m = min(tile_size_m, total_M - cur_start_m)
                    output_range = Slice(tensor_slice=[
                        DimSlice(start=cur_start_n, end=cur_start_n + cur_size_n),
                        DimSlice(start=cur_start_m, end=cur_start_m + cur_size_m),
                        DimSlice(start=0, end=1),
                        DimSlice(start=0, end=1)
                    ])
                    send_inst = Instruction(
                        inst_type=TaskType.SEND,
                        index=shared_index,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        feat_num=1,
                        position=get_list_id(core_x, receiver_y),
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[get_list_id(core_x, sender_y)].insts.append(send_inst)
                    reduce_send_map[(core_x, sender_y)] = send_inst

                    # 2. 生成配对的RECV指令 (使用相同的index)
                    recv_inst = Instruction(
                        inst_type=TaskType.RECV,
                        index=shared_index,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        feat_num=0,
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[get_list_id(core_x, receiver_y)].insts.append(recv_inst)
                    if (core_x, receiver_y) not in reduce_recv_map: reduce_recv_map[(core_x, receiver_y)] = []
                    reduce_recv_map[(core_x, receiver_y)].append(recv_inst)

                    global_inst_id += 1 # 为SEND/RECV对增加一次ID

                    # 2.5 新增：为RECV的数据生成FREE指令，由RECV触发
                    free_inst = Instruction(
                        inst_type=TaskType.FREE,
                        index=global_inst_id,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[get_list_id(core_x, receiver_y)].insts.append(free_inst)
                    recv_inst.trigger_index.append(free_inst.index)
                    global_inst_id += 1

                    # 3. 为接收方生成ELEM指令
                    elem_inst = Instruction(
                        inst_type=TaskType.ELEM,
                        index=global_inst_id,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        feat_num=1, # 简化模型：只依赖自己的部分和
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[get_list_id(core_x, receiver_y)].insts.append(elem_inst)
                    if (core_x, receiver_y) not in reduce_elem_map: reduce_elem_map[(core_x, receiver_y)] = []
                    reduce_elem_map[(core_x, receiver_y)].append(elem_inst)
                    
                    global_inst_id += 1
                
                # 如果需要，由 root core 写回
                if write_back:
                    root_core_id = get_list_id(core_x, root_y)
                    # 计算输出tile的大小
                    cur_start_m = (tp_m * arch_configs.core.x + core_x) * tile_size_m
                    cur_size_m = min(tile_size_m, total_M - cur_start_m)
                    output_range = Slice(tensor_slice=[
                        DimSlice(start=cur_start_n, end=cur_start_n + cur_size_n),
                        DimSlice(start=cur_start_m, end=cur_start_m + cur_size_m),
                        DimSlice(start=0, end=1),
                        DimSlice(start=0, end=1)
                    ])
                    write_inst = Instruction(
                        inst_type=TaskType.WRITE,
                        index=global_inst_id,
                        layer_id=lid,
                        data_type=DataType.FEAT,
                        feat_num=1,
                        tensor_slice=output_range.tensor_slice
                    )
                    pewls[root_core_id].insts.append(write_inst)
                    write_back_inst_map[core_x] = write_inst
                    global_inst_id += 1

                # 4. 建立Reduce阶段的Trigger关系
                for core_y in range(arch_configs.core.y):
                    # 找到该core在k维度累加的最后一个指令
                    last_tp_k = temporal_loop_k - 1
                    final_sum_provider = None
                    if last_tp_k > 0: # Last operation was an ELEM
                        if get_list_id(core_x, core_y) in elem_inst_map and (last_tp_k, tp_m) in elem_inst_map[get_list_id(core_x, core_y)]:
                            final_sum_provider = elem_inst_map[get_list_id(core_x, core_y)][(last_tp_k, tp_m)]
                    else: # Last (and only) operation was an FC
                        if get_list_id(core_x, core_y) in comp_inst_map and (last_tp_k, tp_m) in comp_inst_map[get_list_id(core_x, core_y)]:
                            final_sum_provider = comp_inst_map[get_list_id(core_x, core_y)][(last_tp_k, tp_m)]
                    
                    if not final_sum_provider:
                        continue

                    # A. 触发SEND和ELEM指令 (作为自己的部分和输入)
                    if (core_x, core_y) in reduce_send_map:
                        final_sum_provider.trigger_index.append(reduce_send_map[(core_x, core_y)].index)
                    if (core_x, core_y) in reduce_elem_map:
                        for elem_inst in reduce_elem_map[(core_x, core_y)]:
                            final_sum_provider.trigger_index.append(elem_inst.index)
                
                # C. 写回 (ELEM -> WRITE)
                for core_y in range(arch_configs.core.y):
                    # ELEM 完成后才能进行下一步
                    if (core_x, core_y) in reduce_elem_map:
                        # root core 的 ELEM 需要触发 WRITE
                        if core_y == root_y:
                            if write_back and core_x in write_back_inst_map:
                                for elem in reduce_elem_map[(core_x, core_y)]:
                                    elem.trigger_index.append(write_back_inst_map[core_x].index)
                        # # 非 root core 的 ELEM 不需要再触发 SEND，因为 final_sum_provider 已经并行触发了
                        # else:
                        #     send_inst = reduce_send_map.get((core_x, core_y))
                        #     if send_inst:
                        #         for elem in reduce_elem_map[(core_x, core_y)]:
                        #             elem.trigger_index.append(send_inst.index)
        
        # 统一建立所有计算指令的trigger关系
        for core_id, comp_map_for_core in comp_inst_map.items():
            for (tp_k, tp_m), comp_inst in comp_map_for_core.items():
                core_x, core_y = get_core_coor(core_id)
                # 1. activation
                if read_activation:
                    act_provider = act_providers_map[(core_x, core_y, tp_k)]
                    act_provider.trigger_index.append(comp_inst.index)
                
                # 2. weight
                if read_weight:
                    wgt_provider = wgt_read_map[(core_x, core_y, tp_k, tp_m)]
                    wgt_provider.trigger_index.append(comp_inst.index)
                else:
                    # 如果不读权重，则创建一个LOAD指令来逻辑上提供权重
                    cur_start_k = tp_k * tile_size_k
                    cur_size_k = min(tile_size_k, total_K - cur_start_k)
                    cur_start_m = (tp_m * arch_configs.core.x + core_x) * tile_size_m
                    cur_size_m = min(tile_size_m, total_M - cur_start_m)
                    wgt_range = Slice(tensor_slice=[
                        DimSlice(start=cur_start_m, end=cur_start_m + cur_size_m),
                        DimSlice(start=cur_start_k, end=cur_start_k + cur_size_k)
                    ])
                    wgt_load_inst = Instruction(
                        lid=lid,
                        inst_type=TaskType.LOAD,
                        index=global_inst_id,
                        layer_id=lid,
                        data_type=DataType.PARA,
                        tensor_slice=wgt_range.tensor_slice
                    )
                    pewls[core_id].insts.append(wgt_load_inst)
                    wgt_load_inst.trigger_index.append(comp_inst.index)
                    global_inst_id += 1


        # 建立累加的trigger关系
        for core_id, elem_map_for_core in elem_inst_map.items():
            for (tp_k, tp_m), elem_inst in elem_map_for_core.items():
                # ELEM 指令只在 tp_k > 0 时存在
                
                # 1. 当前的FC指令总是ELEM的一个输入源
                current_fc_inst = comp_inst_map[core_id][(tp_k, tp_m)]
                current_fc_inst.trigger_index.append(elem_inst.index)

                # 2. 另一个输入源是上一个tp_k的累加结果
                prev_tp_k = tp_k - 1
                if prev_tp_k == 0:
                    # 如果是第一个ELEM(tp_k=1)，则依赖于第一个FC(tp_k=0)的结果
                    prev_sum_provider = comp_inst_map[core_id][(prev_tp_k, tp_m)]
                else:
                    # 否则，依赖于前一个ELEM的结果
                    prev_sum_provider = elem_inst_map[core_id][(prev_tp_k, tp_m)]
                
                prev_sum_provider.trigger_index.append(elem_inst.index)
                                    
                                    
if __name__ == "__main__":
    #net = json_analyzer("tools/GEMM.json")

    # 模拟的层ID
    lid = 0
    
    # 模拟一个layer对象，因为GEMM函数需要它
    # 实际使用中，这应该从一个完整的网络定义中获取
    layer = Layer(
        type="fc",
        layer_id=lid,
        layer_group_id=0,
        group_num=1,
        layer_batch_size=64,
        output_partition=Partition(dims=[1, 1, 1, 1]),
        input_fetch=Partition(dims=[1, 1, 1, 1]),
        input_feature=[IFeature(source="dram", source_layer_id=-1, blocks=[])],
        wgt_feature=[],
        output_feature=[]
    )

    # 调用GEMM函数生成指令
    GEMM(read_activation=False, read_weight=False, write_back=True)

    # 将生成的workload输出到json文件
    wl = Workload(name="GEMM_test", pes=pewls)
    workload_json = wl.model_dump_json(indent=4)

    output_path = "../tests/gemm/HW_workload.json"
    with open(output_path, "w") as file:
        print(workload_json, file=file)
    
    print(f"HW_GEMM activation broadcast 指令生成完成！")
    print(f"测试workload已保存至: {output_path}")