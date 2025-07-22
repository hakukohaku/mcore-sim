#!/usr/bin/env python3
"""
路径广播功能测试文件

这个文件演示了如何使用新增的路径广播功能。
"""

from src.sim_type import Message, Instruction, Data, TaskType, DataType, DimSlice

def test_message_broadcast():
    """测试Message类的路径广播功能"""
    
    # 创建一个示例指令
    instruction = Instruction(
        inst_type=TaskType.SEND,
        index=1,
        layer_id=0,
        data_type=DataType.FEAT,
        position=5,  # 目标核心ID
        path_dst=[1, 2, 3],  # 路径广播目标
        tensor_slice=[DimSlice(start=0, end=10)]
    )
    
    # 创建示例数据
    data = Data(index=1, tensor_slice=[DimSlice(start=0, end=10)])
    
    # 创建普通消息（只发送给目标核心）
    normal_msg = Message(
        ins=instruction,
        src=0,
        data=data,
        dst=5
    )
    
    # 创建路径广播消息
    broadcast_msg = Message(
        ins=instruction,
        src=0,
        data=data,
        dst=5,
        path_dst=[1, 2, 3]
    )
    
    # 测试普通消息
    print("=== 普通消息测试 ===")
    print(f"消息目标: {normal_msg.dst}")
    print(f"路径广播目标: {normal_msg.path_dst}")
    
    # 测试各个核心是否应该接收消息
    for core_id in range(6):
        should_receive = normal_msg.should_deliver_to_core(core_id)
        print(f"核心 {core_id} 是否应该接收消息: {should_receive}")
    
    print("\n=== 路径广播消息测试 ===")
    print(f"消息目标: {broadcast_msg.dst}")
    print(f"路径广播目标: {broadcast_msg.path_dst}")
    
    # 测试各个核心是否应该接收消息
    for core_id in range(6):
        should_receive = broadcast_msg.should_deliver_to_core(core_id)
        print(f"核心 {core_id} 是否应该接收消息: {should_receive}")
    
    print("\n=== 测试结果验证 ===")
    # 验证普通消息只有目标核心接收
    assert normal_msg.should_deliver_to_core(5) == True, "目标核心应该接收消息"
    assert normal_msg.should_deliver_to_core(1) == False, "非目标核心不应该接收普通消息"
    
    # 验证广播消息的目标核心和路径核心都接收
    assert broadcast_msg.should_deliver_to_core(5) == True, "目标核心应该接收广播消息"
    assert broadcast_msg.should_deliver_to_core(1) == True, "路径核心1应该接收广播消息"
    assert broadcast_msg.should_deliver_to_core(2) == True, "路径核心2应该接收广播消息"
    assert broadcast_msg.should_deliver_to_core(3) == True, "路径核心3应该接收广播消息"
    assert broadcast_msg.should_deliver_to_core(4) == False, "非相关核心不应该接收广播消息"
    
    print("所有测试通过！路径广播功能正常工作。")

if __name__ == "__main__":
    test_message_broadcast() 