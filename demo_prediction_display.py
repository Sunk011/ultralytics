#!/usr/bin/env python3
"""
卡尔曼滤波预测框显示功能演示脚本

此脚本演示了如何使用增强的 Ultralytics 跟踪器的预测显示功能。
当目标从 Tracked 状态转为 Lost 状态时，会显示卡尔曼滤波预测的边界框。
"""

import numpy as np
import cv2
from argparse import Namespace
import sys
import os

# 添加ultralytics到路径
sys.path.insert(0, '/home/sk/project/ultralytics')

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT

class MockDetectionResults:
    """模拟检测结果类"""
    def __init__(self, bboxes, scores, classes):
        if len(bboxes) == 0:
            self.xywh = np.empty((0, 4), dtype=np.float32)
            self.conf = np.array([], dtype=np.float32)
            self.cls = np.array([], dtype=np.float32)
        else:
            self.xywh = np.array(bboxes, dtype=np.float32)
            self.conf = np.array(scores, dtype=np.float32)
            self.cls = np.array(classes, dtype=np.float32)

def draw_bbox(img, bbox, track_id, score, color, is_prediction=False):
    """在图像上绘制边界框"""
    x1, y1, x2, y2 = bbox[:4].astype(int)
    
    # 预测框使用虚线，真实框使用实线
    line_type = cv2.LINE_4 if is_prediction else cv2.LINE_8
    thickness = 1 if is_prediction else 2
    
    # 绘制边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, line_type)
    
    # 添加标签
    label = f"ID:{int(track_id)} {score:.2f}"
    if is_prediction:
        label += " (PRED)"
    
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                 (x1 + label_size[0], y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def create_demo_scenarios():
    """创建演示场景"""
    scenarios = []
    
    # 场景1: 单个目标短暂丢失
    scenarios.append({
        'name': '单目标短暂丢失',
        'frames': [
            # Frame 1-2: 正常检测
            {'bboxes': [[100, 100, 50, 80]], 'scores': [0.9], 'classes': [0]},
            {'bboxes': [[105, 105, 50, 80]], 'scores': [0.88], 'classes': [0]},
            # Frame 3-7: 目标丢失
            {'bboxes': [], 'scores': [], 'classes': []},
            {'bboxes': [], 'scores': [], 'classes': []},
            {'bboxes': [], 'scores': [], 'classes': []},
            {'bboxes': [], 'scores': [], 'classes': []},
            {'bboxes': [], 'scores': [], 'classes': []},
            # Frame 8: 目标重新出现
            {'bboxes': [[130, 130, 50, 80]], 'scores': [0.87], 'classes': [0]},
        ]
    })
    
    # 场景2: 多目标部分丢失
    scenarios.append({
        'name': '多目标部分丢失',
        'frames': [
            # Frame 1-2: 两个目标都检测到
            {'bboxes': [[100, 100, 50, 80], [200, 150, 60, 90]], 'scores': [0.9, 0.85], 'classes': [0, 1]},
            {'bboxes': [[105, 105, 50, 80], [205, 155, 60, 90]], 'scores': [0.88, 0.83], 'classes': [0, 1]},
            # Frame 3-5: 第一个目标丢失
            {'bboxes': [[210, 160, 60, 90]], 'scores': [0.82], 'classes': [1]},
            {'bboxes': [[215, 165, 60, 90]], 'scores': [0.80], 'classes': [1]},
            {'bboxes': [[220, 170, 60, 90]], 'scores': [0.78], 'classes': [1]},
            # Frame 6: 第一个目标重新出现
            {'bboxes': [[120, 120, 50, 80], [225, 175, 60, 90]], 'scores': [0.86, 0.76], 'classes': [0, 1]},
        ]
    })
    
    return scenarios

def run_demo_scenario(scenario, tracker_type='byte'):
    """运行演示场景"""
    print(f"\n{'='*60}")
    print(f"演示场景: {scenario['name']}")
    print(f"跟踪器类型: {tracker_type.upper()}")
    print('='*60)
    
    # 创建跟踪器参数
    args = Namespace(
        track_high_thresh=0.7,
        track_low_thresh=0.4,
        match_thresh=0.8,
        new_track_thresh=0.6,
        track_buffer=30,
        fuse_score=False
    )
    
    # 添加BOTSORT特定参数
    if tracker_type == 'botsort':
        args.gmc_method = 'sparseOptFlow'
        args.proximity_thresh = 0.5
        args.appearance_thresh = 0.25
        args.with_reid = False
        args.model = 'auto'
        tracker = BOTSORT(args, frame_rate=30)
    else:
        tracker = BYTETracker(args, frame_rate=30)
    
    # 创建画布
    img_width, img_height = 640, 480
    
    # 处理每一帧
    for frame_idx, frame_data in enumerate(scenario['frames'], 1):
        print(f"\n--- Frame {frame_idx} ---")
        
        # 创建检测结果
        detections = MockDetectionResults(
            frame_data['bboxes'],
            frame_data['scores'],
            frame_data['classes']
        )
        
        # 更新跟踪器
        active_tracks = tracker.update(detections)
        all_tracks = tracker.get_all_tracks_for_display()
        
        # 打印结果
        print(f"检测到的目标: {len(frame_data['bboxes'])}")
        print(f"活跃跟踪: {len(active_tracks)}")
        print(f"总显示跟踪: {len(all_tracks)}")
        print(f"预测显示项: {len(tracker.prediction_display)}")
        
        # 详细显示每个跟踪
        for track in all_tracks:
            track_id = int(track[4])
            score = track[5]
            is_prediction = track_id in tracker.prediction_display
            status = "PREDICTION" if is_prediction else "DETECTION"
            print(f"  Track {track_id}: {status}, Score: {score:.3f}")
        
        # 显示预测信息
        for track_id, pred_info in tracker.prediction_display.items():
            remaining = pred_info['remaining_frames']
            print(f"  预测显示 Track {track_id}: 剩余 {remaining} 帧")
        
        # 创建可视化图像
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 50
        
        # 绘制所有跟踪框
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, track in enumerate(all_tracks):
            track_id = int(track[4])
            score = track[5]
            color = colors[track_id % len(colors)]
            is_prediction = track_id in tracker.prediction_display
            
            draw_bbox(img, track, track_id, score, color, is_prediction)
        
        # 添加帧信息
        cv2.putText(img, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Active: {len(active_tracks)}, Total: {len(all_tracks)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 保存图像（可选）
        # cv2.imwrite(f'/tmp/demo_{tracker_type}_{scenario["name"]}_{frame_idx:02d}.jpg', img)
        
        # 显示图像（如果有显示环境）
        # 注释掉显示部分避免无头环境的问题
        # try:
        #     cv2.imshow(f'Demo: {scenario["name"]} - {tracker_type.upper()}', img)
        #     cv2.waitKey(500)  # 500ms延迟
        # except:
        #     pass  # 无显示环境时跳过
    
    # try:
    #     cv2.destroyAllWindows()
    # except:
    #     pass
    
    print(f"\n场景 '{scenario['name']}' 完成!")

def main():
    """主函数"""
    print("卡尔曼滤波预测框显示功能演示")
    print("="*60)
    
    # 创建演示场景
    scenarios = create_demo_scenarios()
    
    # 运行演示
    for scenario in scenarios:
        # 使用BYTETracker演示
        run_demo_scenario(scenario, 'byte')
        
        # 使用BOTSORT演示
        run_demo_scenario(scenario, 'botsort')
    
    print("\n" + "="*60)
    print("所有演示完成!")
    print("="*60)
    
    print("\n关键特性展示:")
    print("✅ 状态监控: Tracked -> Lost 转换检测")
    print("✅ 预测生成: 卡尔曼滤波预测框生成")
    print("✅ 显示管理: 5帧预测显示周期")
    print("✅ 优先级处理: 检测框优先于预测框")
    print("✅ 跟踪器兼容: BYTETracker + BOTSORT")

if __name__ == "__main__":
    main()
