#!/usr/bin/env python3
"""
Test script for prediction display functionality in Ultralytics trackers.
This script tests the feature that displays predicted bounding boxes for 5 frames
after a target transitions from Tracked to Lost state.
"""

import numpy as np
from argparse import Namespace
import sys
import os

# Add ultralytics to path
sys.path.insert(0, '/home/sk/project/ultralytics')

from ultralytics.trackers.byte_tracker import BYTETracker, STrack, TrackState
from ultralytics.trackers.bot_sort import BOTSORT

def create_mock_detection_results(bboxes, scores, classes):
    """Create mock detection results for testing."""
    class MockResults:
        def __init__(self, bboxes, scores, classes):
            if len(bboxes) == 0:
                self.xywh = np.empty((0, 4), dtype=np.float32)
                self.conf = np.array([], dtype=np.float32)
                self.cls = np.array([], dtype=np.float32)
            else:
                self.xywh = np.array(bboxes, dtype=np.float32)  # [x_center, y_center, width, height]
                self.conf = np.array(scores, dtype=np.float32)
                self.cls = np.array(classes, dtype=np.float32)
    
    return MockResults(bboxes, scores, classes)

def test_byte_tracker_prediction_display():
    """Test BYTETracker prediction display functionality."""
    print("Testing BYTETracker prediction display functionality...")
    
    # Create tracker arguments
    args = Namespace(
        track_high_thresh=0.7,
        track_low_thresh=0.4,
        match_thresh=0.8,
        new_track_thresh=0.6,
        track_buffer=30,
        fuse_score=False
    )
    
    # Initialize tracker
    tracker = BYTETracker(args, frame_rate=30)
    
    print(f"Initialized tracker with prediction_frames = {tracker.prediction_frames}")
    
    # Frame 1: Initial detection
    print("\n--- Frame 1: Initial Detection ---")
    detections_1 = create_mock_detection_results(
        bboxes=[[100, 100, 50, 80], [200, 150, 60, 90]],  # Two objects
        scores=[0.9, 0.85],
        classes=[0, 1]
    )
    
    results_1 = tracker.update(detections_1)
    print(f"Frame 1 results: {len(results_1)} active tracks")
    for result in results_1:
        print(f"  Track ID: {result[4]}, Score: {result[5]:.2f}, Class: {result[6]}")
    
    # Check prediction display
    print(f"Prediction display entries: {len(tracker.prediction_display)}")
    
    # Frame 2: Both objects still detected
    print("\n--- Frame 2: Both Objects Detected ---")
    detections_2 = create_mock_detection_results(
        bboxes=[[105, 105, 50, 80], [205, 155, 60, 90]],  # Slight movement
        scores=[0.88, 0.83],
        classes=[0, 1]
    )
    
    results_2 = tracker.update(detections_2)
    print(f"Frame 2 results: {len(results_2)} active tracks")
    for result in results_2:
        print(f"  Track ID: {result[4]}, Score: {result[5]:.2f}, Class: {result[6]}")
    
    print(f"Prediction display entries: {len(tracker.prediction_display)}")
    
    # Frame 3: First object lost (not detected)
    print("\n--- Frame 3: First Object Lost ---")
    detections_3 = create_mock_detection_results(
        bboxes=[[210, 160, 60, 90]],  # Only second object
        scores=[0.82],
        classes=[1]
    )
    
    results_3 = tracker.update(detections_3)
    print(f"Frame 3 results: {len(results_3)} active tracks")
    for result in results_3:
        print(f"  Track ID: {result[4]}, Score: {result[5]:.2f}, Class: {result[6]}")
    
    print(f"Prediction display entries: {len(tracker.prediction_display)}")
    for track_id, pred_info in tracker.prediction_display.items():
        print(f"  Track {track_id}: {pred_info['remaining_frames']} frames remaining")
    
    # Get all tracks including predictions
    all_tracks = tracker.get_all_tracks_for_display()
    print(f"All tracks for display: {len(all_tracks)}")
    for track in all_tracks:
        print(f"  Track ID: {track[4]}, Score: {track[5]:.2f}, BBox: [{track[0]:.1f}, {track[1]:.1f}, {track[2]:.1f}, {track[3]:.1f}]")
    
    # Frame 4-7: Continue without first object to test prediction frames
    for frame_num in range(4, 8):
        print(f"\n--- Frame {frame_num}: First Object Still Missing ---")
        detections = create_mock_detection_results(
            bboxes=[[215, 165, 60, 90]],  # Only second object, continue moving
            scores=[0.80],
            classes=[1]
        )
        
        results = tracker.update(detections)
        print(f"Frame {frame_num} results: {len(results)} active tracks")
        
        print(f"Prediction display entries: {len(tracker.prediction_display)}")
        for track_id, pred_info in tracker.prediction_display.items():
            print(f"  Track {track_id}: {pred_info['remaining_frames']} frames remaining")
        
        # Get all tracks including predictions
        all_tracks = tracker.get_all_tracks_for_display()
        print(f"All tracks for display: {len(all_tracks)}")
        for track in all_tracks:
            print(f"  Track ID: {track[4]}, Score: {track[5]:.2f}")
    
    # Frame 8: First object reappears
    print("\n--- Frame 8: First Object Reappears ---")
    detections_8 = create_mock_detection_results(
        bboxes=[[130, 130, 50, 80], [220, 170, 60, 90]],  # Both objects
        scores=[0.87, 0.79],
        classes=[0, 1]
    )
    
    results_8 = tracker.update(detections_8)
    print(f"Frame 8 results: {len(results_8)} active tracks")
    for result in results_8:
        print(f"  Track ID: {result[4]}, Score: {result[5]:.2f}, Class: {result[6]}")
    
    print(f"Prediction display entries: {len(tracker.prediction_display)}")
    
    print("\nBYTETracker prediction display test completed successfully!")

def test_botsort_prediction_display():
    """Test BOTSORT prediction display functionality."""
    print("\n" + "="*60)
    print("Testing BOTSORT prediction display functionality...")
    
    # Create BOTSORT-specific arguments
    args = Namespace(
        track_high_thresh=0.7,
        track_low_thresh=0.4,
        match_thresh=0.8,
        new_track_thresh=0.6,
        track_buffer=30,
        fuse_score=False,
        gmc_method='sparseOptFlow',
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
        model='auto'
    )
    
    # Initialize BOTSORT tracker
    tracker = BOTSORT(args, frame_rate=30)
    
    print(f"Initialized BOTSORT tracker with prediction_frames = {tracker.prediction_frames}")
    
    # Simple test with one object
    print("\n--- Frame 1: Initial Detection ---")
    detections_1 = create_mock_detection_results(
        bboxes=[[100, 100, 50, 80]],
        scores=[0.9],
        classes=[0]
    )
    
    results_1 = tracker.update(detections_1)
    print(f"Frame 1 results: {len(results_1)} active tracks")
    
    # Frame 2: Object lost
    print("\n--- Frame 2: Object Lost ---")
    detections_2 = create_mock_detection_results(
        bboxes=[],  # No detections
        scores=[],
        classes=[]
    )
    
    results_2 = tracker.update(detections_2)
    print(f"Frame 2 results: {len(results_2)} active tracks")
    print(f"Prediction display entries: {len(tracker.prediction_display)}")
    
    # Get all tracks including predictions
    all_tracks = tracker.get_all_tracks_for_display()
    print(f"All tracks for display: {len(all_tracks)}")
    
    print("\nBOTSORT prediction display test completed successfully!")

if __name__ == "__main__":
    print("Starting Ultralytics Tracker Prediction Display Tests")
    print("="*60)
    
    try:
        test_byte_tracker_prediction_display()
        test_botsort_prediction_display()
        print("\n" + "="*60)
        print("All tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
