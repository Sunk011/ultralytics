#!/usr/bin/env python3
"""
测试从配置文件读取 lost_display_frames 的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_reading():
    """测试从配置文件读取 lost_display_frames"""
    print("测试从配置文件读取 lost_display_frames...")
    
    try:
        from ultralytics.utils import DEFAULT_CFG
        
        # 检查是否能读取到配置项
        lost_display_frames = getattr(DEFAULT_CFG, 'lost_display_frames', None)
        print(f"从配置文件读取到的 lost_display_frames: {lost_display_frames}")
        
        if lost_display_frames is not None:
            print(f"✓ 成功从配置文件读取到值: {lost_display_frames}")
        else:
            print("⚠ 配置项不存在，将使用默认值 5")
            lost_display_frames = 5
        
        # 测试 STrack.mark_lost 方法
        from ultralytics.trackers.byte_tracker import STrack
        from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
        
        # 清空计数器
        STrack.lost_display_counter.clear()
        
        # 创建测试轨迹
        xywh = [100.0, 150.0, 50.0, 75.0, 1]
        score = 0.9
        cls = "person"
        track = STrack(xywh, score, cls)
        
        # 激活轨迹
        kalman_filter = KalmanFilterXYAH()
        track.activate(kalman_filter, 1)
        
        # 测试 mark_lost 方法（不传参数，使用配置文件的值）
        track.mark_lost()
        
        counter_value = STrack.lost_display_counter.get(track.track_id, 0)
        print(f"轨迹 {track.track_id} 在计数器中的值: {counter_value}")
        
        if counter_value == lost_display_frames:
            print("✓ mark_lost 方法正确使用了配置文件中的值")
        else:
            print(f"✗ mark_lost 方法使用的值不正确，期望: {lost_display_frames}, 实际: {counter_value}")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_reading()
