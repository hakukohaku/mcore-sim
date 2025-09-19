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

arch_configs = config_analyzer("../arch/myarch_gemini4_4_cim.json")
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

def get_list_id(x: int, y: int) -> int:
    return x * arch_configs.core.y + y

def test_pipeline(read_activation=True, read_weight=True, write_back=True):
    global global_inst_id, pewls, lid

    # 定义流水线路径和固定的维度
    pipeline_list = [0, 1, 2, 3, 7, 6, 5, 4]
    total_N = 32
    total_K = 48
    total_M = 48

    last_send_recv_id = -1
    
    # --- 定义通用的张量 Slices ---
    # (N, K)
    act_slice = Slice(tensor_slice=[
        DimSlice(start=0, end=total_N),
        DimSlice(start=0, end=total_K)
    ])
    
    # (M, K)
    weight_slice = Slice(tensor_slice=[
        DimSlice(start=0, end=total_M),
        DimSlice(start=0, end=total_K)
    ])

    # (N, M)
    # 因为 K==M, 所以 output_slice 和 act_slice 形状相同
    output_slice = Slice(tensor_slice=[
        DimSlice(start=0, end=total_N),
        DimSlice(start=0, end=total_M)
    ])
    
    # 遍历流水线中的每个阶段
    for i, core_id in enumerate(pipeline_list):
        current_core_id = core_id
        # 计算下一个核心的ID，实现循环流水线
        next_core_id = pipeline_list[(i + 1) % len(pipeline_list)]

        # --- 1. 为输入Activation生成 READ 或 RECV 指令 ---
        if i == 0:
            # 流水线第一级：从DRAM读取初始数据
            input_provider_inst = Instruction(
                inst_type=TaskType.READ,
                index=global_inst_id,
                layer_id=lid,
                data_type=DataType.FEAT,
                tensor_slice=act_slice.tensor_slice
            )
            pewls[current_core_id].insts.append(input_provider_inst)
            global_inst_id += 1
        else:
            # 中间或最后阶段：接收来自上一级的数据
            input_provider_inst = Instruction(
                inst_type=TaskType.RECV,
                index=last_send_recv_id,  # 与上一级的SEND指令共享ID
                layer_id=lid,
                data_type=DataType.FEAT,
                feat_num=0, # RECV指令初始feat_num为0
                tensor_slice=act_slice.tensor_slice
            )
            pewls[current_core_id].insts.append(input_provider_inst)
            # RECV指令不消耗新的global_inst_id

        # --- 2. 为预加载的权重生成 LOAD 指令 ---
        load_wgt_inst = Instruction(
            inst_type=TaskType.LOAD,
            index=global_inst_id,
            layer_id=lid,
            data_type=DataType.PARA,
            tensor_slice=weight_slice.tensor_slice
        )
        pewls[current_core_id].insts.append(load_wgt_inst)
        global_inst_id += 1

        # --- 3. 生成 FC 计算指令 ---
        comp_inst = Instruction(
            inst_type=TaskType.FC,
            index=global_inst_id,
            layer_id=lid,
            data_type=DataType.FEAT,
            feat_num=1,
            para_num=1,
            tensor_slice=output_slice.tensor_slice
        )
        pewls[current_core_id].insts.append(comp_inst)
        
        # 设置依赖：READ/RECV 和 LOAD 都完成后才能开始计算
        input_provider_inst.trigger_index.append(comp_inst.index)
        input_provider_inst.trigger_index.append(load_wgt_inst.index)
        load_wgt_inst.trigger_index.append(comp_inst.index)
        global_inst_id += 1

        # --- 4. 生成 SEND 指令，将结果发送到下一级 ---
        # 为本次SEND/RECV对分配一个新的共享ID
        last_send_recv_id = global_inst_id
        
        send_inst = Instruction(
            inst_type=TaskType.SEND,
            index=last_send_recv_id,
            layer_id=lid,
            data_type=DataType.FEAT,
            feat_num=1,
            position=next_core_id,
            tensor_slice=output_slice.tensor_slice
        )
        pewls[current_core_id].insts.append(send_inst)
        
        # 设置依赖：计算完成后才能发送结果
        comp_inst.trigger_index.append(send_inst.index)
        global_inst_id += 1

    # 循环结束后, last_send_recv_id 保存的是最后一个核心(4)发往第一个核心(0)的SEND指令ID
    # 为核心0补上对应的RECV指令以闭合环路
    core_0_id = pipeline_list[0]
    loop_back_recv_inst = Instruction(
        inst_type=TaskType.RECV,
        index=last_send_recv_id,
        layer_id=lid,
        data_type=DataType.FEAT,
        feat_num=0,
        tensor_slice=output_slice.tensor_slice # 接收的tensor形状与发送时一致
    )
    pewls[core_0_id].insts.append(loop_back_recv_inst)


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

    # 调用函数生成流水线指令
    test_pipeline()

    # 将生成的workload输出到json文件
    wl = Workload(name="pipeline_test", pes=pewls)
    workload_json = wl.model_dump_json(indent=4)

    output_path = "../tests/pipeline/workload_pipeline_fixed_test.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        print(workload_json, file=file)
    
    print(f"Pipeline instructions generated successfully!")
    print(f"Test workload saved to: {output_path}")