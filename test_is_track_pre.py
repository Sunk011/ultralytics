#!/usr/bin/env python3
"""
测试 is_track_pre 属性功能的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

def test_boxes_is_track_pre():
    """测试 Boxes 类的 is_track_pre 属性功能"""
    print("测试 Boxes 类的 is_track_pre 属性功能...")
    
    try:
        from ultralytics.engine.results import Boxes
        
        # 测试1: 不设置 is_track_pre（默认情况）
        print("\n1. 测试默认情况（不设置 is_track_pre）:")
        boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        orig_shape = (480, 640)
        boxes = Boxes(boxes_data, orig_shape)
        
        print(f"   boxes.is_track_pre: {boxes.is_track_pre}")
        assert boxes.is_track_pre is None, "默认情况下 is_track_pre 应该为 None"
        print("   ✓ 默认情况测试通过")
        
        # 测试2: 显式设置 is_track_pre
        print("\n2. 测试显式设置 is_track_pre:")
        is_track_pre = torch.tensor([False, True], dtype=torch.bool)  # 第一个是检测，第二个是预测
        boxes_with_pre = Boxes(boxes_data, orig_shape, is_track_pre)
        
        print(f"   boxes_with_pre.is_track_pre: {boxes_with_pre.is_track_pre}")
        assert boxes_with_pre.is_track_pre is not None, "设置后 is_track_pre 不应该为 None"
        assert torch.equal(boxes_with_pre.is_track_pre, is_track_pre), "is_track_pre 值应该匹配"
        print("   ✓ 显式设置测试通过")
        
        # 测试3: 设备转换
        print("\n3. 测试设备转换:")
        cpu_boxes = boxes_with_pre.cpu()
        print(f"   cpu转换后 is_track_pre: {cpu_boxes.is_track_pre}")
        assert cpu_boxes.is_track_pre is not None, "CPU转换后 is_track_pre 应该保持"
        
        numpy_boxes = boxes_with_pre.numpy()
        print(f"   numpy转换后 is_track_pre: {numpy_boxes.is_track_pre}")
        print(f"   numpy转换后 is_track_pre 类型: {type(numpy_boxes.is_track_pre)}")
        assert numpy_boxes.is_track_pre is not None, "numpy转换后 is_track_pre 应该保持"
        print("   ✓ 设备转换测试通过")
        
        # 测试4: 索引操作
        print("\n4. 测试索引操作:")
        first_box = boxes_with_pre[0]
        print(f"   第一个box的 is_track_pre: {first_box.is_track_pre}")
        assert first_box.is_track_pre is not None, "索引后 is_track_pre 应该保持"
        assert first_box.is_track_pre.item() == False, "第一个box应该是非预测的"
        
        second_box = boxes_with_pre[1]
        print(f"   第二个box的 is_track_pre: {second_box.is_track_pre}")
        assert second_box.is_track_pre.item() == True, "第二个box应该是预测的"
        print("   ✓ 索引操作测试通过")
        
        # 测试5: 兼容性 - 使用numpy数组
        print("\n5. 测试numpy数组兼容性:")
        boxes_data_np = np.array([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        is_track_pre_np = np.array([False, True], dtype=bool)
        boxes_np = Boxes(boxes_data_np, orig_shape, is_track_pre_np)
        
        print(f"   numpy boxes is_track_pre: {boxes_np.is_track_pre}")
        assert boxes_np.is_track_pre is not None, "numpy数组应该支持 is_track_pre"
        print("   ✓ numpy兼容性测试通过")
        
        print("\n所有 Boxes 类测试通过！✅")
        return True
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """测试集成功能"""
    print("\n\n集成测试:")
    print("这个测试验证整个流程是否正常工作")
    
    # 模拟跟踪器返回的数据
    tracks = np.array([[100, 50, 150, 100, 1, 0.9, 0, 0],   # 正常跟踪，最后一列是索引
                      [200, 150, 250, 200, 2, 0.8, 1, 1]])  # 正常跟踪
    
    lost_tmp = np.array([[300, 250, 350, 300, 3, 0.7, 0, 2]])  # 丢失预测
    
    # 模拟 track.py 中的处理逻辑
    combined_data = []
    is_track_pre_flags = []
    
    if len(tracks) > 0:
        combined_data.append(tracks[:, :-1])  # 排除最后一列索引
        is_track_pre_flags.extend([False] * len(tracks))
    
    if len(lost_tmp) > 0:
        combined_data.append(lost_tmp[:, :-1])  # 排除最后一列索引
        is_track_pre_flags.extend([True] * len(lost_tmp))
    
    if len(combined_data) > 0:
        all_tracks = np.vstack(combined_data)
        print(f"   合并后的tracks形状: {all_tracks.shape}")
        print(f"   is_track_pre标识: {is_track_pre_flags}")
        
        # 创建Boxes对象
        from ultralytics.engine.results import Boxes
        orig_shape = (480, 640)
        is_track_pre_tensor = torch.tensor(is_track_pre_flags, dtype=torch.bool)
        
        boxes = Boxes(torch.tensor(all_tracks, dtype=torch.float32), orig_shape, is_track_pre_tensor)
        
        print(f"   创建的boxes数量: {len(boxes.data)}")
        print(f"   is_track_pre: {boxes.is_track_pre}")
        
        # 验证结果
        expected_flags = [False, False, True]  # 前两个是正常跟踪，第三个是预测
        actual_flags = boxes.is_track_pre.tolist()
        
        assert actual_flags == expected_flags, f"期望 {expected_flags}, 实际 {actual_flags}"
        print("   ✓ 集成测试通过")
        return True
    
    return False

if __name__ == "__main__":
    print("开始测试 is_track_pre 功能...\n")
    
    success1 = test_boxes_is_track_pre()
    success2 = test_integration()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！功能实现正确。")
    else:
        print("\n❌ 部分测试失败。")
