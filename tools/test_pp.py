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

class Core(BaseModel):
    x: int
    y: int

class Transformer_Layer(BaseModel):
    # MNK 矩阵乘法示意图 (A[M×N] · B[N×K] → C[M×K])
    # A: 行数 M, 列数 N
    # B: 行数 N, 列数 K
    # C: 行数 M, 列数 K
    #
    #             N                               K
    #          <------>                        <------>
    #        ┌───────────┐                  ┌──────────┐
    #     M  │           │                  │          │
    #        │     A     │       ×          │    B     │
    #        │   (M×K)   │                  │   (N×K)  │
    #        └───────────┘                  └──────────┘
    #                   \\                      //
    #                    \\                    //
    #                     \\                  //
    #                       ┌────────────────┐
    #                    M  │       C        │
    #                       │     (M×K)      │
    #                       └────────────────┘
    name: str
    n_loop: int
    dim_M: int
    dim_N: int
    dim_K: int

class Transformer_Network(BaseModel):
    n_layers: int
    layers: List[Transformer_Layer]

class Pipe_Config(BaseModel):
    core_list: List[int]
    loop: int
    #example: layer=16 pipe_stage=4
    #loop=1: 0-3, 4-7, 8-11, 12-15
    #loop=2: 0-1+8-9, 2-3+10-11, 4-5+12-13, 6-7+14-15

class Transformer_model(BaseModel):
    type: str
    n_layers: int
    n_head: int
    n_kv_head: int
    dim: int
    head_dim: int
    ffn_dim: int
    input_len: int
    batch_size: int
    layers: List=["qkv_gen", "attn_score", "attn_context", "output", "ffn_f1", "ffn_f2"]

def data_mapping(input_model: Transformer_model) -> Transformer_Network:
    network = Transformer_Network(n_layers=input_model.n_layers, layers=[])
    for layer_name in input_model.layers:
        if layer_name == "qkv_gen":
            dim_M = input_model.batch_size * input_model.input_len
            dim_N = input_model.dim
            dim_K = input_model.n_head * input_model.head_dim + 2 * input_model.n_kv_head * input_model.head_dim
            n_loop = 1
        elif layer_name == "attn_score":
            dim_M = input_model.input_len * (input_model.n_head // input_model.n_kv_head) #GQA
            dim_N = input_model.head_dim
            dim_K = input_model.input_len
            n_loop = input_model.batch_size * input_model.n_kv_head
        elif layer_name == "attn_context":
            dim_M = input_model.input_len
            dim_N = input_model.input_len
            dim_K = input_model.head_dim
            n_loop = input_model.batch_size * input_model.n_kv_head
        elif layer_name == "output":
            dim_M = input_model.batch_size * input_model.input_len
            dim_N = input_model.dim
            dim_K = input_model.dim
            n_loop = 1
        elif layer_name == "ffn_f1":
            dim_M = input_model.batch_size * input_model.input_len
            dim_N = input_model.dim
            dim_K = input_model.ffn_dim
            n_loop = 1
        elif layer_name == "ffn_f2":
            dim_M = input_model.batch_size * input_model.input_len
            dim_N = input_model.ffn_dim
            dim_K = input_model.dim
            n_loop = 1
        network.layers.append(Transformer_Layer(name=layer_name, dim_N=dim_N, dim_K=dim_K, dim_M=dim_M, n_loop=n_loop))
    return network



def layer_cmd_gen_for_single_core( network: Transformer_Network, core_id: int, tf_layer_id: int, next_core_id: int, pipe_recv_en: False, pipe_sent_en: False, finish: False):
    global global_inst_id, last_send_id, pewls

    for layer_id, layer in enumerate(network.layers):
        act_slice = Slice(tensor_slice=[
            DimSlice(start=0, end=layer.dim_M),
            DimSlice(start=0, end=layer.dim_N)
        ])
        #####
        weight_slice = Slice(tensor_slice=[
            DimSlice(start=0, end=layer.dim_K),
            DimSlice(start=0, end=layer.dim_N)
        ])
        #####
        output_slice = Slice(tensor_slice=[
            DimSlice(start=0, end=layer.dim_M),
            DimSlice(start=0, end=layer.dim_K)
        ])

        for layer_loop_id in range(layer.n_loop):
    # --- 1. 为输入Activation生成 READ / RECV / LOAD 指令 ---
            if layer_id == 0: #第一层
                if tf_layer_id == 0: #第一层
                    # 模型第一层：从DRAM读取初始数据
                    input_provider_inst = Instruction(
                        inst_type=TaskType.READ,
                        inst_name="READ",
                        index=global_inst_id,
                        layer_loop_id=layer_loop_id,
                        layer_id=tf_layer_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        tensor_slice=act_slice.tensor_slice
                    )
                elif pipe_recv_en:
                    # 模型第一层但有流水线数据：接收上一级流水线core的数据
                    input_provider_inst = Instruction(
                        inst_type=TaskType.RECV,
                        inst_name="RECV",
                        index=last_send_id,  # 与上一级的SEND指令共享ID
                        layer_loop_id=layer_loop_id,
                        layer_id=tf_layer_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        feat_num=0, # RECV指令初始feat_num为0
                        tensor_slice=act_slice.tensor_slice
                    )
                else:
                    #模型第一层且无流水线数据传递：从本地SRAM加载数据
                    input_provider_inst = Instruction(
                        inst_type=TaskType.LOAD,
                        inst_name="LOAD",
                        index=global_inst_id,  
                        layer_loop_id=layer_loop_id,
                        layer_id=tf_layer_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        feat_num=0, # RECV指令初始feat_num为0
                        tensor_slice=act_slice.tensor_slice
                    )
            else:
                #非模型第一层：从本地SRAM加载数据
                input_provider_inst = Instruction(
                    inst_type=TaskType.LOAD,
                    inst_name="LOAD_IN",
                    index=global_inst_id,  
                    layer_id=tf_layer_id,
                    layer_name=layer.name,
                    data_type=DataType.FEAT,
                    feat_num=0, # RECV指令初始feat_num为0
                    tensor_slice=act_slice.tensor_slice
                )
            pewls[core_id].insts.append(input_provider_inst)
            global_inst_id += 1
                # RECV指令不消耗新的global_inst_id

            # --- 2. 为预加载的权重生成 LOAD 指令 ---
            load_wgt_inst = Instruction(
                inst_type=TaskType.LOAD,
                inst_name="LOAD_W",
                index=global_inst_id,
                layer_loop_id=layer_loop_id,
                layer_id=tf_layer_id,
                layer_name=layer.name,
                data_type=DataType.PARA,
                tensor_slice=weight_slice.tensor_slice
            )
            pewls[core_id].insts.append(load_wgt_inst)
            global_inst_id += 1

            # --- 3. 生成 FC 计算指令 ---
            comp_inst = Instruction(
                inst_type=TaskType.FC,
                inst_name="FC",
                index=global_inst_id,
                layer_loop_id=layer_loop_id,
                layer_id=tf_layer_id,
                layer_name=layer.name,
                data_type=DataType.FEAT,
                feat_num=1,
                para_num=1,
                tensor_slice=output_slice.tensor_slice
            )
            pewls[core_id].insts.append(comp_inst)
            
            # 设置依赖：READ/RECV 和 LOAD 都完成后才能开始计算
            input_provider_inst.trigger_index.append(comp_inst.index)
            input_provider_inst.trigger_index.append(load_wgt_inst.index)
            load_wgt_inst.trigger_index.append(comp_inst.index)
            global_inst_id += 1

            # --- 4. 生成 SEND/STORE 指令，将结果发送到下一级/存在本地 ---
            # 为本次SEND/RECV对分配一个新的共享ID
            last_send_id = global_inst_id
            if (layer_id == len(network.layers) - 1) and pipe_sent_en:
                #模型最后一层且有流水线数据传递：结果发送给下一级流水线core
                result_inst = Instruction(
                    inst_type=TaskType.SEND,
                    inst_name="SEND",
                    index=global_inst_id,
                    layer_loop_id=layer_loop_id,
                    layer_id=tf_layer_id,
                    layer_name=layer.name,
                    data_type=DataType.FEAT,
                    feat_num=1,
                    position=next_core_id,
                    tensor_slice=output_slice.tensor_slice
                )

            else:
                #非模型最后一层或无流水线数据传递：结果存在本地SRAM
                result_inst = Instruction(
                    inst_type=TaskType.STORE,
                    inst_name="STORE",
                    index=global_inst_id,
                    layer_loop_id=layer_loop_id,
                    layer_id=tf_layer_id,
                    layer_name=layer.name,
                    data_type=DataType.FEAT,
                    feat_num=1,
                    tensor_slice=output_slice.tensor_slice
                )
            pewls[core_id].insts.append(result_inst)
            
            # 设置依赖：计算完成后才能发送结果
            comp_inst.trigger_index.append(result_inst.index)
            global_inst_id += 1

            # 循环结束后, last_send_recv_id 保存的是最后一个核心(4)发往第一个核心(0)的SEND指令ID
            # 为核心0补上对应的RECV指令以闭合环路
            if layer_id == len(network.layers) - 1 and finish:
                loop_back_recv_inst = Instruction(
                    inst_type=TaskType.RECV,
                    inst_name="RECV",
                    index=last_send_id,
                    layer_loop_id=layer_loop_id,
                    layer_id=tf_layer_id,
                    layer_name=layer.name,
                    data_type=DataType.FEAT,
                    feat_num=0,
                    tensor_slice=output_slice.tensor_slice # 接收的tensor形状与发送时一致
                )
                pewls[next_core_id].insts.append(loop_back_recv_inst)


def cmd_stream_gen(tf_network: Transformer_Network, pipeline_config: Pipe_Config):

    pipe_stage = len(pipeline_config.core_list)
    layer_per_loop = tf_network.n_layers // pipeline_config.loop
    layer_per_loop_stage = tf_network.n_layers // pipeline_config.loop // pipe_stage
    rece_inst_id = [0 for id in range(pipe_stage)]

    for loop_id in range(pipeline_config.loop):            
        for i, core_id in enumerate(pipeline_config.core_list):
            for layer_id in range(layer_per_loop_stage):
                if i < pipe_stage - 1:
                    next_core_id = pipeline_config.core_list[i+1]
                else:
                    next_core_id = pipeline_config.core_list[0]
                    
                pipe_recv_en = (layer_id == 0)
                pipe_sent_en = (layer_id == layer_per_loop_stage - 1)
                finish = (loop_id == pipeline_config.loop - 1) and (layer_id == layer_per_loop_stage - 1) and (i == pipe_stage - 1)
                tf_layer_id = loop_id * layer_per_loop + i *layer_per_loop_stage + layer_id
                print(f"tf_layer_id: {tf_layer_id}",f"core_id: {core_id}",f"next_core_id: {next_core_id}",f"pipe_recv_en: {pipe_recv_en}",f"pipe_sent_en: {pipe_sent_en}",f"finish: {finish}")
                layer_cmd_gen_for_single_core(tf_network, core_id, tf_layer_id, next_core_id, pipe_recv_en, pipe_sent_en, finish)
            

if __name__ == "__main__":

    arch_configs = config_analyzer("/home/syhe/mcore-sim/arch/myarch_gemini4_4_cim.json")
    arch_configs.core.spm.size /= 4
    core_num = arch_configs.core.x * arch_configs.core.y

    inf = 100000
    global_inst_id = 0
    last_send_recv_id = 0
    pewls = [PEworkload(id=id) for id in range(core_num)]

    transformer_model = Transformer_model(
        type="transformer",
        n_layers=16,
        n_head=1,
        n_kv_head=1,
        dim=32,
        head_dim=32,
        ffn_dim=64,
        input_len=32,
        batch_size=1,
        layers=["qkv_gen", "attn_score", "attn_context", "output", "ffn_f1", "ffn_f2"]
    )
    tf_network = data_mapping(transformer_model)
    pipeline_config = Pipe_Config(core_list=[0, 1, 2, 3, 7, 6, 5, 4], loop=2)
    cmd_stream_gen(tf_network, pipeline_config)


    # 将生成的workload输出到json文件
    wl = Workload(name="pipeline_test", pes=pewls)
    workload_json = wl.model_dump_json(indent=4)

    output_path = "../tests/pipeline/workload_pipeline_fixed_test.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        print(workload_json, file=file)
    
    print(f"Pipeline instructions generated successfully!")
    print(f"Test workload saved to: {output_path}")