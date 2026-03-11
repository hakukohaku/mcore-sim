import os
import sys
import math
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import Dict, List
import argparse
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

def json_analyzer(filename: str):
    with open(filename, 'r') as file:
        data = json.load(file)
        return data

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
    nonlinear_en: bool
    dim_M: int
    dim_N: int
    dim_K: int

class Transformer_Network(BaseModel):
    n_blocks: int
    layers: List[Transformer_Layer]

class Pipe_Config(BaseModel):
    core_list: List[int]
    loop: int
    #example: layer=16 pipe_stage=4
    #loop=1: 0-3, 4-7, 8-11, 12-15
    #loop=2: 0-1+8-9, 2-3+10-11, 4-5+12-13, 6-7+14-15

class Transformer_model(BaseModel):
    type: str
    n_blocks: int
    n_head: int
    n_kv_head: int
    dim: int
    head_dim: int
    ffn_dim: int
    input_len: int
    batch_size: int
    layers: List=["qkv_gen", "attn_score", "attn_context", "output", "ffn_f1", "ffn_f2"]


def load_pipeline_config(config: dict, core_num: int) -> Pipe_Config:
    pipeline_config_data = config.get("pipeline_config")
    if pipeline_config_data is None:
        raise ValueError("Config must contain 'pipeline_config'.")

    pipeline_config = Pipe_Config.model_validate(pipeline_config_data)
    if not pipeline_config.core_list:
        raise ValueError("Pipeline core_list must not be empty.")
    if pipeline_config.loop <= 0:
        raise ValueError("Pipeline loop must be a positive integer.")
    if len(set(pipeline_config.core_list)) != len(pipeline_config.core_list):
        raise ValueError("Pipeline core_list must not contain duplicate core ids.")
    if min(pipeline_config.core_list) < 0 or max(pipeline_config.core_list) >= core_num:
        raise ValueError(f"Pipeline core ids must be in [0, {core_num - 1}].")
    return pipeline_config


def data_mapping(input_model: Transformer_model) -> Transformer_Network:
    network = Transformer_Network(n_blocks=input_model.n_blocks, layers=[])
    for layer_name in input_model.layers:
        if layer_name == "qkv_gen":
            dim_M = input_model.batch_size * input_model.input_len
            dim_N = input_model.dim
            dim_K = input_model.n_head * input_model.head_dim + 2 * input_model.n_kv_head * input_model.head_dim
            n_loop = 1
            nonlinear_en = False
        elif layer_name == "attn_score":
            dim_M = input_model.input_len * (input_model.n_head // input_model.n_kv_head) #GQA
            dim_N = input_model.head_dim
            dim_K = input_model.input_len
            n_loop = input_model.batch_size * input_model.n_kv_head
            nonlinear_en = True
        elif layer_name == "attn_context":
            dim_M = input_model.input_len
            dim_N = input_model.input_len
            dim_K = input_model.head_dim
            n_loop = input_model.batch_size * input_model.n_kv_head
            nonlinear_en = False
        elif layer_name == "output":
            dim_M = input_model.batch_size * input_model.input_len
            dim_N = input_model.dim
            dim_K = input_model.dim
            n_loop = 1
            nonlinear_en = True
        elif layer_name == "ffn_f1":
            dim_M = input_model.batch_size * input_model.input_len
            dim_N = input_model.dim
            dim_K = input_model.ffn_dim
            n_loop = 1
            nonlinear_en = True
        elif layer_name == "ffn_f2":
            dim_M = input_model.batch_size * input_model.input_len
            dim_N = input_model.ffn_dim
            dim_K = input_model.dim
            n_loop = 1
            nonlinear_en = True
        network.layers.append(Transformer_Layer(name=layer_name, dim_N=dim_N, dim_K=dim_K, dim_M=dim_M, n_loop=n_loop, nonlinear_en=nonlinear_en))
    return network



def layer_cmd_gen_for_single_core(network: Transformer_Network, core_id: int, block_id: int, last_core_id: int, next_core_id: int, n_micro_batch: int, pipe_recv_en: False, pipe_sent_en: False, finish: False):
    global global_inst_id, pewls, last_send_id
    
    for micro_batch_id in range(n_micro_batch):

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
                if layer_id == 0: # block 内第一层算子
                    if block_id == 0: # 第一个 Transformer block
                        # 模型第一个 block：从DRAM读取初始数据
                        if micro_batch_id == 0:
                            input_provider_inst = Instruction(
                                inst_type=TaskType.READ,
                                inst_name="READ",
                                index=global_inst_id,
                                layer_loop_id=layer_loop_id,
                                layer_id=block_id,
                                layer_name=layer.name,
                                data_type=DataType.FEAT,
                                feat_num=0,
                                tensor_slice=act_slice.tensor_slice
                            )
                        else:
                            input_provider_inst = Instruction(
                                inst_type=TaskType.READ,
                                inst_name="READ",
                                index=global_inst_id,
                                layer_loop_id=layer_loop_id,
                                layer_id=block_id,
                                layer_name=layer.name,
                                data_type=DataType.FEAT,
                                feat_num=1,
                                tensor_slice=act_slice.tensor_slice
                            )
                            op_finsih_inst.trigger_index.append(input_provider_inst.index)

                    elif pipe_recv_en:
                        # block 内第一层算子且有流水线数据：接收上一级流水线 core 的数据
                        input_provider_inst = Instruction(
                            inst_type=TaskType.RECV,
                            inst_name="RECV",
                            index=last_send_id[micro_batch_id][last_core_id],  # 与上一级的SEND指令共享ID
                            layer_loop_id=layer_loop_id,
                            layer_id=block_id,
                            layer_name=layer.name,
                            data_type=DataType.FEAT,
                            feat_num=1, # RECV指令初始feat_num为0
                            tensor_slice=act_slice.tensor_slice
                        )
                        
                    else:
                        # block 内第一层算子且无流水线数据传递：从本地 SRAM 加载数据
                        input_provider_inst = Instruction(
                            inst_type=TaskType.LOAD,
                            inst_name="LOAD",
                            index=global_inst_id,  
                            layer_loop_id=layer_loop_id,
                            layer_id=block_id,
                            layer_name=layer.name,
                            data_type=DataType.FEAT,
                            feat_num=1, 
                            tensor_slice=act_slice.tensor_slice
                        )
                        op_finsih_inst.trigger_index.append(input_provider_inst.index)
                else:
                    # block 内非第一层算子：从本地 SRAM 加载数据
                    input_provider_inst = Instruction(
                        inst_type=TaskType.LOAD,
                        inst_name="LOAD_IN",
                        index=global_inst_id,  
                        layer_id=block_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        feat_num=1, 
                        tensor_slice=act_slice.tensor_slice
                    )
                    op_finsih_inst.trigger_index.append(input_provider_inst.index)
                pewls[core_id].insts.append(input_provider_inst)
                global_inst_id += 1
                    
                # --- 2. 为预加载的权重生成 LOAD 指令 ---
                load_wgt_inst = Instruction(
                    inst_type=TaskType.LOAD,
                    inst_name="LOAD_W",
                    index=global_inst_id,
                    layer_loop_id=layer_loop_id,
                    layer_id=block_id,
                    layer_name=layer.name,
                    data_type=DataType.PARA,
                    feat_num=1,
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
                    layer_id=block_id,
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


                if layer.nonlinear_en:

                    nonlinear_inst = Instruction(
                        inst_type=TaskType.NON_LINEAR,
                        inst_name="NONLINEAR",
                        index=global_inst_id,
                        layer_loop_id=layer_loop_id,
                        layer_id=block_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        feat_num=1,
                        tensor_slice=output_slice.tensor_slice
                    )

                    pewls[core_id].insts.append(nonlinear_inst)
                    comp_inst.trigger_index.append(nonlinear_inst.index)
                    global_inst_id += 1

                # --- 4. 生成 SEND/STORE 指令，将结果发送到下一级/存在本地 ---
                # 为本次SEND/RECV对分配一个新的共享ID
                last_send_id[micro_batch_id][core_id] = global_inst_id
                if (layer_id == len(network.layers) - 1) and pipe_sent_en:
                    #模型最后一层且有流水线数据传递：结果发送给下一级流水线core
                    result_inst = Instruction(
                        inst_type=TaskType.SEND,
                        inst_name="SEND",
                        index=global_inst_id,
                        layer_loop_id=layer_loop_id,
                        layer_id=block_id,
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
                        layer_id=block_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        feat_num=1,
                        tensor_slice=output_slice.tensor_slice
                    )
                    op_finsih_inst = result_inst

                pewls[core_id].insts.append(result_inst)
                
                # 设置依赖：计算完成后才能发送结果
                if layer.nonlinear_en:
                    nonlinear_inst.trigger_index.append(result_inst.index)
                else:
                    comp_inst.trigger_index.append(result_inst.index)
                
                #comp_inst.trigger_index.append(result_inst.index)
                global_inst_id += 1

                # 循环结束后, last_send_recv_id 保存的是最后一个核心(4)发往第一个核心(0)的SEND指令ID
                # 为核心0补上对应的RECV指令以闭合环路
                if layer_id == len(network.layers) - 1 and finish:
                    loop_back_recv_inst = Instruction(
                        inst_type=TaskType.RECV,
                        inst_name="RECV",
                        index=last_send_id[micro_batch_id][core_id],
                        layer_loop_id=layer_loop_id,
                        layer_id=block_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        feat_num=1,
                        tensor_slice=output_slice.tensor_slice # 接收的tensor形状与发送时一致
                    )
                    pewls[next_core_id].insts.append(loop_back_recv_inst)


def cmd_stream_gen(network: Transformer_Network, n_micro_batch: int, pipeline_config: Pipe_Config):

    pipe_stage = len(pipeline_config.core_list)
    block_per_loop = network.n_blocks // pipeline_config.loop
    block_per_stage = network.n_blocks // pipeline_config.loop // pipe_stage

    for loop_id in range(pipeline_config.loop):            
        for i, core_id in enumerate(pipeline_config.core_list):
            for block_offset in range(block_per_stage):
                if i < pipe_stage - 1:
                    next_core_id = pipeline_config.core_list[i+1]
                else:
                    next_core_id = pipeline_config.core_list[0]

                if i > 0:
                    last_core_id = pipeline_config.core_list[i-1]
                else:
                    last_core_id = pipeline_config.core_list[pipe_stage - 1]    

                pipe_recv_en = (block_offset == 0)
                pipe_sent_en = (block_offset == block_per_stage - 1)
                finish = (loop_id == pipeline_config.loop - 1) and (block_offset == block_per_stage - 1) and (i == pipe_stage - 1)
                block_id = loop_id * block_per_loop + i * block_per_stage + block_offset

                
                #print(f"block_id: {block_id}",f"core_id: {core_id}",f"next_core_id: {next_core_id}",f"pipe_recv_en: {pipe_recv_en}",f"pipe_sent_en: {pipe_sent_en}",f"finish: {finish}")
                layer_cmd_gen_for_single_core(network, core_id, block_id, last_core_id, next_core_id, n_micro_batch, pipe_recv_en, pipe_sent_en, finish)
            

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int,help='batch size')
    parser.add_argument('-a', '--architecture', type=str, help='architecture')
    parser.add_argument('-mb', '--micro_batch', type=int, help='micro batch size')
    parser.add_argument('-dp', '--dp', type=int, help='data parallelism')
    parser.add_argument('-o', '--output', type=str, help='output path')
    parser.add_argument('-c', '--config', type=str, help='config')
    args = parser.parse_args()

    batch = args.batch
    micro_batch = args.micro_batch
    dp = args.dp
    output_file = args.output
    architecture = args.architecture
    arch_configs = config_analyzer(architecture)
    arch_configs.core.spm.size /= 4
    core_num = arch_configs.core.x * arch_configs.core.y
    config = json_analyzer(args.config)
    model = config["model"]
    pipeline_config = load_pipeline_config(config, core_num)

    channel = model["n_channel"]
    
    inf = 10000000
    global_inst_id = 0
    pewls = [PEworkload(id=id) for id in range(core_num)]
    batch_per_dp = math.ceil(batch * channel / dp)
    n_micro_batch = math.ceil(batch_per_dp / micro_batch)
    last_send_id = [[[] for id in range(core_num)] for id in range(n_micro_batch)]

    transformer_model = Transformer_model(
        type="transformer",
        n_blocks=model["n_blocks"],
        n_head=model["n_head"],
        n_kv_head=model["n_kv_head"],
        dim=model["dim"],
        head_dim=model["head_dim"],
        ffn_dim=model["ffn_dim"],
        input_len=model["input_len"],
        batch_size=micro_batch,
        layers=["qkv_gen", "attn_score", "attn_context", "output", "ffn_f1", "ffn_f2"]
    )
    network = data_mapping(transformer_model)
    cmd_stream_gen(network, n_micro_batch, pipeline_config)


    # 将生成的workload输出到json文件
    wl = Workload(name="inst_stream", pes=pewls)
    workload_json = wl.model_dump_json(indent=4)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as file:
        print(workload_json, file=file)
    
    print(f"Pipeline instructions generated successfully!")
    print(f"Test workload saved to: {output_file}")
