#!/usr/bin/env python3
"""
is_track_pre 功能使用示例
演示如何使用新增的跟踪预测标识功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    from ultralytics.engine.results import Boxes
    
    # 创建检测框数据
    boxes_data = torch.tensor([
        [100, 50, 150, 100, 0.9, 0],   # 第1个框：实际检测
        [200, 150, 300, 250, 0.8, 1],  # 第2个框：实际检测  
        [300, 200, 400, 300, 0.7, 0],  # 第3个框：跟踪预测
    ])
    
    # 创建跟踪预测标识
    is_track_pre = torch.tensor([False, False, True])  # 第3个框是预测
    
    # 创建Boxes对象
    orig_shape = (480, 640)
    boxes = Boxes(boxes_data, orig_shape, is_track_pre)
    
    print(f"检测框数量: {len(boxes.data)}")
    print(f"坐标 (xyxy): {boxes.xyxy}")
    print(f"置信度: {boxes.conf}")
    print(f"类别: {boxes.cls}")
    print(f"跟踪预测标识: {boxes.is_track_pre}")
    
    # 筛选出预测框
    prediction_masks = boxes.is_track_pre
    if prediction_masks is not None:
        prediction_indices = torch.where(prediction_masks)[0]
        print(f"预测框的索引: {prediction_indices}")
        
        if len(prediction_indices) > 0:
            prediction_boxes = boxes[prediction_indices]
            print(f"预测框坐标: {prediction_boxes.xyxy}")
            print(f"预测框置信度: {prediction_boxes.conf}")

def example_track_integration():
    """跟踪集成示例"""
    print("\n=== 跟踪集成示例 ===")
    print("模拟跟踪器返回数据的处理过程")
    
    # 模拟跟踪器返回的数据
    tracks = np.array([
        [100, 50, 150, 100, 1, 0.9, 0, 0],   # 正常跟踪1
        [200, 150, 250, 200, 2, 0.8, 1, 1],  # 正常跟踪2
    ])
    
    lost_tmp = np.array([
        [300, 250, 350, 300, 3, 0.7, 0, 2],  # 丢失预测1
        [400, 350, 450, 400, 4, 0.6, 1, 3],  # 丢失预测2
    ])
    
    print(f"正常跟踪数量: {len(tracks)}")
    print(f"丢失预测数量: {len(lost_tmp)}")
    
    # 模拟track.py中的处理逻辑
    combined_data = []
    is_track_pre_flags = []
    
    # 添加正常跟踪数据
    if len(tracks) > 0:
        combined_data.append(tracks[:, :-1])  # 排除最后一列索引
        is_track_pre_flags.extend([False] * len(tracks))
    
    # 添加丢失预测数据
    if len(lost_tmp) > 0:
        combined_data.append(lost_tmp[:, :-1])  # 排除最后一列索引
        is_track_pre_flags.extend([True] * len(lost_tmp))
    
    # 合并数据
    all_tracks = np.vstack(combined_data)
    print(f"合并后总数量: {len(all_tracks)}")
    print(f"跟踪预测标识: {is_track_pre_flags}")
    
    # 创建Boxes对象
    from ultralytics.engine.results import Boxes
    orig_shape = (480, 640)
    is_track_pre_tensor = torch.tensor(is_track_pre_flags, dtype=torch.bool)
    
    boxes = Boxes(torch.tensor(all_tracks, dtype=torch.float32), orig_shape, is_track_pre_tensor)
    
    print(f"最终Boxes对象:")
    print(f"  - 总框数: {len(boxes.data)}")
    print(f"  - 实际检测数: {(~boxes.is_track_pre).sum().item()}")
    print(f"  - 跟踪预测数: {boxes.is_track_pre.sum().item()}")
    
    return boxes

def example_filtering_visualization():
    """筛选和可视化示例"""
    print("\n=== 筛选和可视化示例 ===")
    
    # 使用前面的例子获取boxes
    boxes = example_track_integration()
    
    # 分别处理实际检测和跟踪预测
    if boxes.is_track_pre is not None:
        # 筛选实际检测框
        detection_mask = ~boxes.is_track_pre
        detection_boxes = boxes[detection_mask]
        
        # 筛选跟踪预测框
        prediction_mask = boxes.is_track_pre
        prediction_boxes = boxes[prediction_mask]
        
        print(f"实际检测框:")
        for i, (coord, conf, cls) in enumerate(zip(detection_boxes.xyxy, detection_boxes.conf, detection_boxes.cls)):
            print(f"  框{i+1}: 坐标{coord.tolist()}, 置信度{conf:.3f}, 类别{int(cls)}")
        
        print(f"跟踪预测框:")
        for i, (coord, conf, cls) in enumerate(zip(prediction_boxes.xyxy, prediction_boxes.conf, prediction_boxes.cls)):
            print(f"  框{i+1}: 坐标{coord.tolist()}, 置信度{conf:.3f}, 类别{int(cls)}")
        
        # 可视化信息
        print(f"\n可视化建议:")
        print(f"  - 实际检测框: 使用实线边框，正常颜色")
        print(f"  - 跟踪预测框: 使用虚线边框，半透明颜色")

def example_device_operations():
    """设备操作示例"""
    print("\n=== 设备操作示例 ===")
    
    from ultralytics.engine.results import Boxes
    
    # 创建示例数据
    boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
    is_track_pre = torch.tensor([False, True])
    orig_shape = (480, 640)
    
    boxes = Boxes(boxes_data, orig_shape, is_track_pre)
    print(f"原始数据类型: {type(boxes.data)}")
    print(f"原始is_track_pre类型: {type(boxes.is_track_pre)}")
    
    # CPU操作
    cpu_boxes = boxes.cpu()
    print(f"CPU后数据类型: {type(cpu_boxes.data)}")
    print(f"CPU后is_track_pre: {cpu_boxes.is_track_pre}")
    
    # NumPy操作
    numpy_boxes = boxes.numpy()
    print(f"NumPy后数据类型: {type(numpy_boxes.data)}")
    print(f"NumPy后is_track_pre类型: {type(numpy_boxes.is_track_pre)}")
    print(f"NumPy后is_track_pre: {numpy_boxes.is_track_pre}")
    
    # 索引操作
    first_box = boxes[0]
    print(f"索引后第一个框的is_track_pre: {first_box.is_track_pre}")

if __name__ == "__main__":
    print("is_track_pre 功能使用示例\n")
    
    try:
        example_basic_usage()
        example_track_integration() 
        example_filtering_visualization()
        example_device_operations()
        
        print("\n🎉 所有示例运行成功！")
        print("\n使用建议:")
        print("1. 在跟踪模式下，可以通过 result.boxes.is_track_pre 访问预测标识")
        print("2. True 表示跟踪预测框，False 表示实际检测框")
        print("3. 可以根据这个标识进行不同的可视化处理")
        print("4. 普通检测模式下，is_track_pre 为 None，不影响现有功能")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()
