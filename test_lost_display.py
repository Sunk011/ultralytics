#!/usr/bin/env python3
"""
测试 lost 显示计数器功能的简单脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics.trackers.byte_tracker import STrack, BYTETracker
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
import numpy as np

def test_lost_display_counter():
    """测试 lost 显示计数器功能"""
    print("测试 STrack lost 显示计数器功能...")
    
    # 清空计数器
    STrack.lost_display_counter.clear()
    
    # 创建一个测试轨迹
    xywh = [100.0, 150.0, 50.0, 75.0, 1]  # x, y, w, h, idx
    score = 0.9
    cls = "person"
    track = STrack(xywh, score, cls)
    
    # 激活轨迹
    kalman_filter = KalmanFilterXYAH()
    track.activate(kalman_filter, 1)
    print(f"轨迹 {track.track_id} 已激活，状态: {track.state}")
    
    # 检查初始状态
    assert len(STrack.lost_display_counter) == 0, "初始状态下计数器应该为空"
    print("✓ 初始状态检查通过")
    
    # 标记为 lost
    display_frames = 5
    track.mark_lost(display_frames)
    print(f"轨迹 {track.track_id} 被标记为 lost，状态: {track.state}")
    
    # 检查计数器
    assert track.track_id in STrack.lost_display_counter, "轨迹ID应该在计数器中"
    assert STrack.lost_display_counter[track.track_id] == display_frames, f"计数器应该为 {display_frames}"
    assert STrack.should_display_lost_track(track.track_id), "应该显示 lost 轨迹"
    print("✓ mark_lost 功能检查通过")
    
    # 测试计数器递减
    for i in range(display_frames):
        print(f"帧 {i+1}: 计数器 = {STrack.lost_display_counter[track.track_id]}")
        should_display_before = STrack.should_display_lost_track(track.track_id)
        STrack.update_lost_display_counters()
        should_display_after = STrack.should_display_lost_track(track.track_id)
        
        if i < display_frames - 1:
            assert track.track_id in STrack.lost_display_counter, f"帧 {i+1}: 轨迹ID应该还在计数器中"
            print(f"    帧 {i+1}: 轨迹仍需显示")
        else:
            assert track.track_id not in STrack.lost_display_counter, f"帧 {i+1}: 轨迹ID应该已从计数器中移除"
            print(f"    帧 {i+1}: 轨迹不再显示")
    
    print("✓ 计数器递减功能检查通过")
    
    # 测试重新激活
    STrack.lost_display_counter[track.track_id] = 3  # 手动添加回去
    new_track = STrack([105.0, 155.0, 55.0, 80.0, 1], 0.95, "person")
    track.re_activate(new_track, 10)
    
    assert track.track_id not in STrack.lost_display_counter, "重新激活后轨迹ID应该从计数器中移除"
    assert track.state == TrackState.Tracked, "重新激活后状态应该为 Tracked"
    print("✓ re_activate 功能检查通过")
    
    print("\n所有测试通过！✅")

if __name__ == "__main__":
    test_lost_display_counter()
