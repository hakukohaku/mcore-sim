#!/usr/bin/env python3
"""
Wormhole路由使用示例

这个示例展示如何使用新实现的Wormhole路由功能
"""

import simpy
import logging
from src.noc_new import NoC, Router
from src.arch_config import NoCConfig, RouterConfig, LinkConfig
from src.sim_type import Message, Data, DataType, Instruction, Record, DimSlice

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_sample_message(env, data_index, dst_router):
    """创建一个示例消息"""
    # 创建一个简单的指令记录
    record = Record()
    
    # 示例张量切片
    tensor_slice = [DimSlice(start=0, end=8), DimSlice(start=0, end=8), DimSlice(start=0, end=8)]
    
    instruction = Instruction(
        index=data_index,
        inst_type=2,  # SEND
        data_type=DataType.FEAT,
        record=record,
        layer_id=0,  # 添加缺失的 layer_id
        tensor_slice=tensor_slice  # 添加缺失的 tensor_slice
    )
    
    # 创建数据
    data = Data(index=data_index, tensor_slice=tensor_slice)
    
    # 创建消息
    message = Message(
        ins=instruction,
        data=data,
        dst=dst_router,
        src=0,
        feat_precision=1,
        para_precision=1
    )
    
    return message

def wormhole_routing_demo():
    """Wormhole路由演示"""
    print("=== Wormhole Routing Demo ===")
    
    # 创建仿真环境
    env = simpy.Environment()
    
    # 创建NoC配置
    link_config = LinkConfig(width=64, delay=1)
    router_config = RouterConfig(type="XY", vc=1)
    noc_config = NoCConfig(type="mesh", x=4, y=4, router=router_config, link=link_config)
    
    # 创建NoC网络
    noc = NoC(env, noc_config)
    noc.build_connection()
    
    print(f"创建了一个 {noc_config.x}x{noc_config.y} 的NoC网络")
    print(f"总共有 {len(noc.routers)} 个路由器")
    
    # 创建示例消息
    src_router_id = 0   # Router(0,0)
    dst_router_id = 15  # Router(3,3)
    message = create_sample_message(env, data_index=1001, dst_router=dst_router_id)
    
    print(f"\n--- 测试路径规划 ---")
    src_router = noc.routers[src_router_id]
    path, directions, hops = src_router.full_path_route(dst_router_id)
    print(f"从Router{src_router_id}到Router{dst_router_id}的路径: {path}")
    print(f"方向序列: {directions}")
    print(f"总跳数: {hops}")
    
    def test_wormhole_routing():
        """测试Wormhole路由的仿真过程"""
        print(f"\n--- 开始Wormhole路由仿真 ---")
        print(f"时间 {env.now}: 开始从Router{src_router_id}发送消息到Router{dst_router_id}")
        
        # 使用Wormhole路由发送消息
        yield env.process(noc.wormhole_send(src_router_id, message))
        
        print(f"时间 {env.now}: Wormhole路由完成")
    
    def test_router_interface():
        """测试Router接口的Wormhole路由"""
        print(f"\n--- 测试Router接口 ---")
        src_router = noc.routers[src_router_id]
        
        # 创建另一个消息
        message2 = create_sample_message(env, data_index=1002, dst_router=dst_router_id)
        
        print(f"时间 {env.now}: 通过Router接口开始Wormhole路由")
        yield env.process(src_router.wormhole_route_via_noc(message2))
        print(f"时间 {env.now}: Router接口Wormhole路由完成")
    
    def test_strategy_comparison():
        """比较不同路由策略"""
        print(f"\n--- 测试路由策略比较 ---")
        src_router = noc.routers[src_router_id]
        
        # 测试Wormhole路由
        message3 = create_sample_message(env, data_index=1003, dst_router=dst_router_id)
        print(f"时间 {env.now}: 开始Wormhole策略路由")
        yield env.process(src_router.route_with_strategy(message3, strategy="wormhole"))
        wormhole_time = env.now
        
        # 重置时间进行比较（实际使用中不会这样做）
        print(f"Wormhole路由完成时间: {wormhole_time}")
    
    # 运行测试
    #env.process(test_wormhole_routing())
    #nv.process(test_router_interface())
    env.process(test_strategy_comparison())
    
    # 运行仿真
    print(f"\n开始仿真...")
    env.run(until=1000)  # 运行1000个时间单位
    
    print(f"\n=== 仿真完成 ===")
    print(f"最终仿真时间: {env.now}")

if __name__ == "__main__":
    wormhole_routing_demo() 