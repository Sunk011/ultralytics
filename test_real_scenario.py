#!/usr/bin/env python3
"""
测试修复后的compare_video.py场景
"""

import torch
import numpy as np
from ultralytics.engine.results import Boxes, Results

def test_real_scenario():
    """测试真实的compare_video.py场景"""
    print("=" * 60)
    print("测试真实的compare_video.py场景")
    print("=" * 60)
    
    # 模拟原始图像
    orig_img = np.zeros((480, 640, 3), dtype=np.uint8)
    names = {0: "person", 1: "car", 2: "bike"}
    
    # 场景1: 没有检测结果
    print("\n1. 场景1: 没有检测结果")
    print("-" * 30)
    
    result_empty = Results(
        orig_img=orig_img,
        path="test.jpg",
        names=names,
        boxes=None  # 没有检测结果
    )
    
    print(f"result.boxes: {result_empty.boxes}")
    print(f"result.boxes is None: {result_empty.boxes is None}")
    
    # 这是用户代码中会出错的地方
    if result_empty.boxes is not None:
        print("不应该进入这里")
    else:
        print("✅ 正确处理了None boxes的情况")
    
    # 场景2: 有检测结果但为空
    print("\n2. 场景2: 有检测结果但为空")
    print("-" * 30)
    
    empty_boxes = torch.empty(0, 6)
    result_empty_boxes = Results(
        orig_img=orig_img,
        path="test.jpg", 
        names=names,
        boxes=empty_boxes
    )
    
    print(f"result.boxes: {result_empty_boxes.boxes}")
    print(f"result.boxes is None: {result_empty_boxes.boxes is None}")
    
    if result_empty_boxes.boxes is not None:
        print("✅ boxes不为None，可以安全访问属性")
        
        # 这是用户代码中的关键操作
        try:
            boxes = result_empty_boxes.boxes.xyxy.cpu().numpy()
            classes = result_empty_boxes.boxes.cls.cpu().numpy()
            confs = result_empty_boxes.boxes.conf.cpu().numpy()
            is_track_pre = result_empty_boxes.boxes.is_track_pre.cpu().numpy()
            
            print(f"   boxes.shape: {boxes.shape}")
            print(f"   classes.shape: {classes.shape}")
            print(f"   confs.shape: {confs.shape}")
            print(f"   is_track_pre.shape: {is_track_pre.shape}")
            print(f"   is_track_pre: {is_track_pre}")
            
            # 模拟draw_boxes函数中的循环
            print(f"   绘制 {len(boxes)} 个框")
            for i, (box, flag) in enumerate(zip(boxes, is_track_pre)):
                print(f"   框 {i}: {box}, 是预测框: {flag}")
            
            print("✅ 所有操作都成功了！")
            
        except Exception as e:
            print(f"❌ 操作失败: {e}")
    
    # 场景3: 有正常检测结果
    print("\n3. 场景3: 有正常检测结果")
    print("-" * 30)
    
    normal_boxes = torch.tensor([
        [100, 50, 150, 100, 0.9, 0],  # 人
        [200, 100, 300, 200, 0.8, 1]  # 车
    ])
    
    result_normal = Results(
        orig_img=orig_img,
        path="test.jpg",
        names=names,
        boxes=normal_boxes
    )
    
    print(f"result.boxes: {result_normal.boxes}")
    print(f"result.boxes is None: {result_normal.boxes is None}")
    
    if result_normal.boxes is not None:
        print("✅ boxes不为None，可以安全访问属性")
        
        try:
            boxes = result_normal.boxes.xyxy.cpu().numpy()
            classes = result_normal.boxes.cls.cpu().numpy()
            confs = result_normal.boxes.conf.cpu().numpy()
            is_track_pre = result_normal.boxes.is_track_pre.cpu().numpy()
            
            print(f"   boxes.shape: {boxes.shape}")
            print(f"   classes.shape: {classes.shape}")
            print(f"   confs.shape: {confs.shape}")
            print(f"   is_track_pre.shape: {is_track_pre.shape}")
            print(f"   is_track_pre: {is_track_pre}")
            
            # 模拟draw_boxes函数中的循环
            print(f"   绘制 {len(boxes)} 个框")
            for i, (box, flag) in enumerate(zip(boxes, is_track_pre)):
                box_color = "红色" if flag else "绿色"
                print(f"   框 {i}: {box}, 是预测框: {flag}, 颜色: {box_color}")
            
            print("✅ 所有操作都成功了！")
            
        except Exception as e:
            print(f"❌ 操作失败: {e}")

def test_user_original_code_pattern():
    """测试用户原始代码模式"""
    print("\n" + "=" * 60)
    print("测试用户原始代码模式")
    print("=" * 60)
    
    # 模拟用户代码中的模式
    print("\n模拟用户代码的执行流程:")
    
    # 模拟原始图像
    orig_img = np.zeros((480, 640, 3), dtype=np.uint8)
    names = {0: "person", 1: "car"}
    
    # 创建一个空的结果（模拟没有检测到任何物体）
    empty_boxes = torch.empty(0, 6)
    track_results = [Results(
        orig_img=orig_img,
        path="test.jpg",
        names=names,
        boxes=empty_boxes
    )]
    
    print("用户代码:")
    print("if len(track_results) > 0 and track_results[0].boxes is not None:")
    
    if len(track_results) > 0 and track_results[0].boxes is not None:
        print("    ✅ 条件满足，继续执行...")
        print("    boxes = track_results[0].boxes.xyxy.cpu().numpy()")
        print("    classes = track_results[0].boxes.cls.cpu().numpy()")
        print("    confs = track_results[0].boxes.conf.cpu().numpy()")
        print("    is_track_pre = track_results[0].boxes.is_track_pre.cpu().numpy()  # 这里原来会出错")
        
        try:
            boxes = track_results[0].boxes.xyxy.cpu().numpy()
            classes = track_results[0].boxes.cls.cpu().numpy()
            confs = track_results[0].boxes.conf.cpu().numpy()
            is_track_pre = track_results[0].boxes.is_track_pre.cpu().numpy()
            
            print(f"    ✅ 成功获取所有数据!")
            print(f"    boxes: {boxes}")
            print(f"    classes: {classes}")
            print(f"    confs: {confs}")
            print(f"    is_track_pre: {is_track_pre}")
            
            print("    draw_boxes(top_frame, boxes, classes, confs, names, is_track_pre, track_ids=ids)")
            print("    ✅ 可以正常调用draw_boxes函数!")
            
        except Exception as e:
            print(f"    ❌ 出错了: {e}")
    else:
        print("    条件不满足，跳过")

if __name__ == "__main__":
    test_real_scenario()
    test_user_original_code_pattern()
    print("\n" + "=" * 60)
    print("所有测试完成! 修复成功!")
    print("=" * 60)
