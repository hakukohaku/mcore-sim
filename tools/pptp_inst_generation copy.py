import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pydantic import BaseModel, ValidationError

from src import ArchConfig
from src.sim_type import DataType, DimSlice, Instruction, PEworkload, Slice, TaskType, Workload


def config_analyzer(filename: str) -> ArchConfig:
    with open(filename, "r") as file:
        data = json.load(file)
        try:
            return ArchConfig.model_validate(data)
        except ValidationError as exc:
            print(exc.json())
            raise


def json_analyzer(filename: str):
    with open(filename, "r") as file:
        return json.load(file)


class TPConfig(BaseModel):
    core_list: List[int]
    tp: int
    n_input_loop: int = 1


class TransformerLayer(BaseModel):
    name: str
    n_loop: int
    nonlinear_en: bool
    reduce_en: bool = False
    dim_M: int
    dim_N: int
    dim_K: int


class TransformerNetwork(BaseModel):
    n_blocks: int
    layers: List[TransformerLayer]


class TransformerModel(BaseModel):
    type: str
    n_blocks: int
    n_head: int
    n_kv_head: int
    n_encoder_kv_len: int = 0
    dim: int
    head_dim: int
    ffn_dim: int
    input_len: int
    batch_size: int
    layers: List[str] = [
        "qkv_gen",
        "attn_score",
        "attn_context",
        "output",
        "ffn_f1",
        "ffn_f2",
    ]


def load_tp_config(config: dict, core_num: int) -> TPConfig:
    tp_config_data = config.get("tp_config")
    if tp_config_data is None:
        raise ValueError("Config must contain 'tp_config'.")

    tp_config = TPConfig.model_validate(tp_config_data)
    if not tp_config.core_list:
        raise ValueError("Tensor parallel core_list must not be empty.")
    if tp_config.n_input_loop <= 0:
        raise ValueError("Tensor parallel n_input_loop must be a positive integer.")
    if len(set(tp_config.core_list)) != len(tp_config.core_list):
        raise ValueError("Tensor parallel core_list must not contain duplicate core ids.")
    if min(tp_config.core_list) < 0 or max(tp_config.core_list) >= core_num:
        raise ValueError(f"Tensor parallel core ids must be in [0, {core_num - 1}].")
    return tp_config

def build_tp_networks(input_model: TransformerModel, tp_config: TPConfig) -> Dict[int, TransformerNetwork]:
    networks: Dict[int, TransformerNetwork] = {}
    tp_degree = tp_config.tp
    
    print(f"tp_degree: {tp_degree}, core_list: {tp_config.core_list}")
    for core_position, core_id in enumerate(tp_config.core_list):
        network = TransformerNetwork(n_blocks=input_model.n_blocks, layers=[])

        def append_layer(name: str, dim_m: int, dim_n: int, dim_k: int, n_loop: int, nonlinear_en: bool, reduce_en: bool = False):
            network.layers.append(
                TransformerLayer(
                    name=name,
                    dim_M=dim_m,
                    dim_N=dim_n,
                    dim_K=dim_k,
                    n_loop=n_loop,
                    nonlinear_en=nonlinear_en,
                    reduce_en=reduce_en
                )
            )

        append_layer(
            name="qkv_gen",
            dim_m=input_model.batch_size * input_model.input_len,
            dim_n=input_model.dim,
            dim_k=(input_model.n_head * input_model.head_dim + 2 * input_model.n_kv_head * input_model.head_dim) // tp_degree,
            n_loop=1,
            nonlinear_en=False,
            reduce_en=True,
        )
        append_layer(
            name="attn_score",
            dim_m=input_model.input_len * (input_model.n_head // input_model.n_kv_head),
            dim_n=input_model.head_dim * input_model.n_kv_head,
            dim_k=(input_model.input_len + input_model.n_encoder_kv_len) // tp_degree,
            n_loop=input_model.batch_size * input_model.n_kv_head,
            nonlinear_en=True,
            reduce_en=True,
        )
        append_layer(
            name="attn_context",
            dim_m=(input_model.input_len + input_model.n_encoder_kv_len),
            dim_n=(input_model.input_len + input_model.n_encoder_kv_len),
            dim_k=input_model.head_dim * input_model.n_kv_head // tp_degree,
            n_loop=input_model.batch_size * input_model.n_kv_head,
            nonlinear_en=False,
            reduce_en=True,
        )
        append_layer(
            name="output",
            dim_m=input_model.batch_size * input_model.input_len,
            dim_n=input_model.dim,
            dim_k=input_model.dim // tp_degree,
            n_loop=1,
            nonlinear_en=True,
            reduce_en=True,
        )
        append_layer(
            name="ffn_f1",
            dim_m=input_model.batch_size * input_model.input_len,
            dim_n=input_model.dim ,
            dim_k=input_model.ffn_dim // tp_degree,
            n_loop=1,
            nonlinear_en=True,
                
        )
        append_layer(
            name="ffn_f2",
            dim_m=input_model.batch_size * input_model.input_len,
            dim_n=input_model.ffn_dim,
            dim_k=input_model.dim // tp_degree ,
            n_loop=1,
            nonlinear_en=True,
            reduce_en=True,
        )
        networks[core_id] = network

    return networks


def build_slices(layer: TransformerLayer) -> Tuple[Slice, Slice, Slice]:
    act_slice = Slice(
        tensor_slice=[
            DimSlice(start=0, end=layer.dim_M),
            DimSlice(start=0, end=layer.dim_N),
        ]
    )
    weight_slice = Slice(
        tensor_slice=[
            DimSlice(start=0, end=layer.dim_K),
            DimSlice(start=0, end=layer.dim_N),
        ]
    )
    output_slice = Slice(
        tensor_slice=[
            DimSlice(start=0, end=layer.dim_M),
            DimSlice(start=0, end=layer.dim_K),
        ]
    )
    return act_slice, weight_slice, output_slice


def scaled_tensor_slice(tensor_slice: List[DimSlice], scale: float) -> List[DimSlice]:
    if scale <= 0 or not tensor_slice:
        return [DimSlice(start=dim.start, end=dim.start) for dim in tensor_slice]

    scaled = [DimSlice(start=dim.start, end=dim.end) for dim in tensor_slice]
    last_dim = scaled[-1]
    dim_len = max(0, last_dim.end - last_dim.start)
    scaled_len = math.floor(dim_len * scale)
    scaled[-1] = DimSlice(start=last_dim.start, end=last_dim.start + scaled_len)
    return scaled


def next_inst_index() -> int:
    global global_inst_id
    index = global_inst_id
    global_inst_id += 1
    return index


def append_instruction(core_id: int, **kwargs) -> Instruction:
    inst = Instruction(**kwargs)
    pewls[core_id].insts.append(inst)
    return inst


def build_feature_provider(
    core_id: int,
    previous_finish_inst: Instruction,
    block_id: int,
    layer_loop_id: int,
    layer: TransformerLayer,
    act_slice: Slice,
) -> Instruction:
    is_first_model_input = previous_finish_inst is None
    if block_id == 0 and layer.name == "qkv_gen":
        input_inst = append_instruction(
            core_id,
            inst_type=TaskType.READ,
            index=next_inst_index(),
            layer_loop_id=layer_loop_id,
            layer_id=block_id,
            layer_name=layer.name,
            data_type=DataType.FEAT,
            feat_num=0 if is_first_model_input else 1,
            tensor_slice=act_slice.tensor_slice,
        )
    else:
        input_inst = append_instruction(
            core_id,
            inst_type=TaskType.LOAD,
            index=next_inst_index(),
            layer_loop_id=layer_loop_id,
            layer_id=block_id,
            layer_name=layer.name,
            data_type=DataType.FEAT,
            feat_num=1,
            tensor_slice=act_slice.tensor_slice,
        )

    if previous_finish_inst is not None:
        previous_finish_inst.trigger_index.append(input_inst.index)
    return input_inst


def layer_cmd_gen_for_single_core(network: TransformerNetwork, core_id: int, n_micro_batch: int, tp_degree: int, n_input_loop: int = 1):
    previous_finish_inst = None
    for m_batch_id in range(n_micro_batch):
        for block_id in range(network.n_blocks):
            for layer in network.layers:
                act_slice, weight_slice, output_slice = build_slices(layer)
                for layer_loop_id in range(layer.n_loop):
                    feature_inst = build_feature_provider(
                        core_id=core_id,
                        previous_finish_inst=previous_finish_inst,
                        block_id=block_id,
                        layer_loop_id=layer_loop_id,
                        layer=layer,
                        act_slice=act_slice,
                    )

                    weight_inst = append_instruction(
                        core_id,
                        inst_type=TaskType.LOAD,
                        index=next_inst_index(),
                        layer_loop_id=layer_loop_id,
                        layer_id=block_id,
                        layer_name=layer.name,
                        data_type=DataType.PARA,
                        tensor_slice=weight_slice.tensor_slice,
                    )

                    comp_inst = append_instruction(
                        core_id,
                        inst_type=TaskType.FC,
                        index=next_inst_index(),
                        layer_loop_id=layer_loop_id,
                        layer_id=block_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        feat_num=1,
                        para_num=1,
                        tensor_slice=output_slice.tensor_slice,
                    )
                    feature_inst.trigger_index.append(comp_inst.index)
                    weight_inst.trigger_index.append(comp_inst.index)

                    result_inst = comp_inst

                    if layer.reduce_en:
                        ring_inst = append_instruction(
                            core_id,
                            inst_type=TaskType.RING,
                            index=next_inst_index(),
                            layer_loop_id=layer_loop_id,
                            layer_id=block_id,
                            layer_name=layer.name,
                            data_type=DataType.FEAT,
                            feat_num=1,
                            tensor_slice=scaled_tensor_slice(
                                output_slice.tensor_slice,
                                (tp_degree - 1) / tp_degree,
                            ),
                        )
                        result_inst.trigger_index.append(ring_inst.index)
                        result_inst = ring_inst

                    if layer.nonlinear_en:
                        nonlinear_inst = append_instruction(
                            core_id,
                            inst_type=TaskType.NON_LINEAR,
                            index=next_inst_index(),
                            layer_loop_id=layer_loop_id,
                            layer_id=block_id,
                            layer_name=layer.name,
                            data_type=DataType.FEAT,
                            feat_num=1,
                            tensor_slice=output_slice.tensor_slice,
                        )
                        result_inst.trigger_index.append(nonlinear_inst.index)
                        result_inst = nonlinear_inst

                    store_inst = append_instruction(
                        core_id,
                        inst_type=TaskType.STORE,
                        index=next_inst_index(),
                        layer_loop_id=layer_loop_id,
                        layer_id=block_id,
                        layer_name=layer.name,
                        data_type=DataType.FEAT,
                        feat_num=1,
                        tensor_slice=output_slice.tensor_slice,
                    )
                    result_inst.trigger_index.append(store_inst.index)
                    previous_finish_inst = store_inst


def cmd_stream_gen(input_model: TransformerModel, n_micro_batch: int, tp_config: TPConfig):
    tp_networks = build_tp_networks(input_model, tp_config)
    for core_id in tp_config.core_list:
        layer_cmd_gen_for_single_core(
            tp_networks[core_id],
            core_id,
            n_micro_batch,
            tp_degree=tp_config.tp,
            n_input_loop=tp_config.n_input_loop,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, help="batch size")
    parser.add_argument("-a", "--architecture", type=str, help="architecture")
    parser.add_argument("-mb", "--micro-batch", type=int, help="micro batch size")
    parser.add_argument("-pp", "--pp", type=int, default=1, help="pipeline parallelism")
    parser.add_argument("-dp", "--dp", type=int, help="data parallelism")
    parser.add_argument("-ekv", "--encoder_kv_len", type=int, default=0, help="encoder KV length")
    parser.add_argument("-o", "--output", type=str, help="output path")
    parser.add_argument("-c", "--config", type=str, help="config")
    args = parser.parse_args()

    batch = args.batch
    dp = args.dp
    pp = args.pp
    micro_batch = args.micro_batch
    output_file = args.output
    architecture = args.architecture
    n_encoder_kv_len = args.encoder_kv_len

    arch_configs = config_analyzer(architecture)
    core_num = arch_configs.core.x * arch_configs.core.y

    config = json_analyzer(args.config)
    model = config["model"]
    tp_config = load_tp_config(config, core_num)

    channel = model["n_channel"]

    global_inst_id = 0
    pewls = [PEworkload(id=idx) for idx in range(core_num)]

    #batch_per_dp = math.ceil(batch * channel / dp)
    batch_per_dp = math.ceil(batch * channel)
    n_micro_batch = math.ceil(batch_per_dp / micro_batch)

    transformer_model = TransformerModel(
        type="transformer",
        n_blocks=model["n_blocks"] // pp,
        n_head=model["n_head"],
        n_kv_head=model["n_kv_head"],
        n_encoder_kv_len=n_encoder_kv_len,
        dim=model["dim"],
        head_dim=model["head_dim"],
        ffn_dim=model["ffn_dim"],
        input_len=model["input_len"],
        batch_size=micro_batch,
    )

    cmd_stream_gen(transformer_model, n_micro_batch, tp_config)

    workload = Workload(name="inst_stream", pes=pewls)
    workload_json = workload.model_dump_json(indent=4)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as file:
        print(workload_json, file=file)

    print("Tensor-parallel instructions generated successfully!")
    print(f"Test workload saved to: {output_file}")
