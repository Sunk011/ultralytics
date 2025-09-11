#!/usr/bin/env python3
"""
测试空检测结果时is_track_pre属性的行为
"""

import torch
import numpy as np
from ultralytics.engine.results import Boxes

def test_empty_boxes():
    """测试空Boxes对象的is_track_pre行为"""
    print("=" * 60)
    print("测试空Boxes对象的is_track_pre行为")
    print("=" * 60)
    
    # 创建空的boxes数据
    empty_boxes_torch = torch.empty(0, 6)  # 0个框，6个特征
    empty_boxes_numpy = np.empty((0, 6))
    orig_shape = (480, 640)
    
    # 测试PyTorch版本
    print("\n1. 测试PyTorch空张量:")
    boxes_torch = Boxes(empty_boxes_torch, orig_shape)
    print(f"   boxes.data.shape: {boxes_torch.data.shape}")
    print(f"   boxes.is_track_pre: {boxes_torch.is_track_pre}")
    print(f"   boxes.is_track_pre.shape: {boxes_torch.is_track_pre.shape}")
    print(f"   boxes.is_track_pre.dtype: {boxes_torch.is_track_pre.dtype}")
    print(f"   boxes.is_track_pre.device: {boxes_torch.is_track_pre.device}")
    
    # 测试设备转换
    print("\n   测试设备转换方法:")
    try:
        cpu_result = boxes_torch.is_track_pre.cpu().numpy()
        print(f"   is_track_pre.cpu().numpy(): {cpu_result}")
        print(f"   类型: {type(cpu_result)}, 形状: {cpu_result.shape}")
    except Exception as e:
        print(f"   ❌ cpu().numpy() 失败: {e}")
    
    # 测试NumPy版本
    print("\n2. 测试NumPy空数组:")
    boxes_numpy = Boxes(empty_boxes_numpy, orig_shape)
    print(f"   boxes.data.shape: {boxes_numpy.data.shape}")
    print(f"   boxes.is_track_pre: {boxes_numpy.is_track_pre}")
    print(f"   boxes.is_track_pre.shape: {boxes_numpy.is_track_pre.shape}")
    print(f"   boxes.is_track_pre.dtype: {boxes_numpy.is_track_pre.dtype}")

def test_normal_boxes():
    """测试正常Boxes对象的is_track_pre行为"""
    print("\n" + "=" * 60)
    print("测试正常Boxes对象的is_track_pre行为（作为对照）")
    print("=" * 60)
    
    # 创建正常的boxes数据
    normal_boxes = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 100, 250, 150, 0.8, 1]])
    orig_shape = (480, 640)
    
    print("\n1. 不带is_track_pre的正常Boxes:")
    boxes = Boxes(normal_boxes, orig_shape)
    print(f"   boxes.data.shape: {boxes.data.shape}")
    print(f"   boxes.is_track_pre: {boxes.is_track_pre}")
    print(f"   boxes.is_track_pre.shape: {boxes.is_track_pre.shape}")
    
    # 测试设备转换
    try:
        cpu_result = boxes.is_track_pre.cpu().numpy()
        print(f"   is_track_pre.cpu().numpy(): {cpu_result}")
    except Exception as e:
        print(f"   ❌ cpu().numpy() 失败: {e}")
    
    print("\n2. 带is_track_pre的正常Boxes:")
    is_track_pre = torch.tensor([False, True])
    boxes_with_pred = Boxes(normal_boxes, orig_shape, is_track_pre)
    print(f"   boxes.data.shape: {boxes_with_pred.data.shape}")
    print(f"   boxes.is_track_pre: {boxes_with_pred.is_track_pre}")
    print(f"   boxes.is_track_pre.shape: {boxes_with_pred.is_track_pre.shape}")
    
    try:
        cpu_result = boxes_with_pred.is_track_pre.cpu().numpy()
        print(f"   is_track_pre.cpu().numpy(): {cpu_result}")
    except Exception as e:
        print(f"   ❌ cpu().numpy() 失败: {e}")

def test_compare_video_scenario():
    """模拟compare_video.py中的场景"""
    print("\n" + "=" * 60)
    print("模拟compare_video.py中的场景")
    print("=" * 60)
    
    # 模拟没有检测结果的情况
    print("\n测试场景: 没有检测结果时的处理")
    
    # 创建空的检测结果
    empty_boxes = torch.empty(0, 6)
    orig_shape = (480, 640)
    boxes = Boxes(empty_boxes, orig_shape)
    
    print(f"模拟: track_results[0].boxes = {boxes}")
    print(f"模拟: track_results[0].boxes is not None = {boxes is not None}")
    
    # 这应该能正常工作，不会报错
    try:
        is_track_pre = boxes.is_track_pre.cpu().numpy()
        print(f"✅ is_track_pre.cpu().numpy() 成功: {is_track_pre}")
        print(f"   类型: {type(is_track_pre)}")
        print(f"   形状: {is_track_pre.shape}")
        print(f"   数据类型: {is_track_pre.dtype}")
        
        # 测试在draw_boxes函数中的使用
        print(f"\n测试在draw_boxes中的使用:")
        print(f"   len(boxes.xyxy.cpu().numpy()): {len(boxes.xyxy.cpu().numpy())}")
        print(f"   len(is_track_pre): {len(is_track_pre)}")
        print(f"   zip(boxes, is_track_pre) 应该为空迭代器")
        
        # 模拟draw_boxes中的循环
        for i, (box, flag) in enumerate(zip(boxes.xyxy.cpu().numpy(), is_track_pre)):
            print(f"   迭代 {i}: box={box}, flag={flag}")
        print("   循环完成（应该没有迭代）")
        
    except Exception as e:
        print(f"❌ 失败: {e}")

if __name__ == "__main__":
    test_empty_boxes()
    test_normal_boxes()
    test_compare_video_scenario()
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
