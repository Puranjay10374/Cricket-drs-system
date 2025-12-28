"""
Cricket DRS - Manual Ball Selection Demo

Demonstrates manual ball selection without using the API server.
Just run this script directly on your video file.
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from drs.tracker import BallTracker
from drs.detectors import AdaptiveBallDetector
from drs.config import BallDetectionConfig, create_stump_region_for_frame, TrajectoryConfig, DecisionConfig
from drs.trajectory import TrajectoryAnalyzer
from drs.decision import DecisionMaker
import cv2


def demo_manual_selection(video_path: str):
    """
    Demo manual ball selection on a video file
    
    Args:
        video_path: Path to cricket video
    """
    print("\n" + "="*70)
    print("🏏 Cricket DRS - Manual Ball Selection Demo")
    print("="*70)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"\n❌ Error: Video file not found: {video_path}")
        return
    
    # Get video dimensions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n❌ Error: Could not open video: {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"\n📹 Video: {video_path}")
    print(f"   Size: {frame_width}x{frame_height}")
    
    # Auto-configure stump region
    stump_region = create_stump_region_for_frame(frame_width, frame_height)
    print(f"\n🎯 Stump Region (auto-configured):")
    print(f"   X: {stump_region.x:.0f}")
    print(f"   Y: {stump_region.y_top:.0f} to {stump_region.y_bottom:.0f}")
    
    # Initialize components
    print("\n⚙️  Initializing tracker...")
    detector = AdaptiveBallDetector()  # Learns from manual selection!
    tracker = BallTracker(
        detector=detector,
        config=BallDetectionConfig(),
        use_kalman=True,
        use_manual_selection=True  # ← KEY: Enable manual selection
    )
    
    print("\n🎯 Manual Selection Instructions:")
    print("   1. A window will show the first frame")
    print("   2. Click on the cricket ball")
    print("   3. Press ENTER to confirm")
    print("   4. Press ESC to cancel (uses automatic detection)")
    print("\n" + "-"*70)
    
    # Track ball (will prompt for manual selection)
    tracking_result = tracker.track_video(video_path, frame_skip=1)
    
    # Display results
    print("\n" + "="*70)
    print("📊 TRACKING RESULTS")
    print("="*70)
    print(f"Total frames: {tracking_result['total_frames']}")
    print(f"Frames tracked: {tracking_result['frames_tracked']}")
    print(f"Tracking quality: {tracking_result['tracking_quality']:.1f}%")
    print(f"FPS: {tracking_result['fps']}")
    
    if tracking_result['frames_tracked'] > 0:
        # Analyze trajectory
        analyzer = TrajectoryAnalyzer()
        trajectory_analysis = analyzer.analyze_trajectory(
            tracking_result['trajectory'],
            stump_region
        )
        
        # Make decision
        decision_maker = DecisionMaker()
        decision_result = decision_maker.make_decision(
            trajectory_analysis,
            tracking_result
        )
        
        print("\n" + "="*70)
        print("⚖️  DRS DECISION")
        print("="*70)
        print(f"Decision: {decision_result['decision']}")
        print(f"Confidence: {decision_result['confidence']:.1f}%")
        if decision_result.get('impact_point'):
            impact = decision_result['impact_point']
            print(f"Impact point: ({impact[0]:.1f}, {impact[1]:.1f})")
        if decision_result.get('reasoning'):
            print(f"Reasoning: {decision_result['reasoning']}")
        
        print("\n" + "="*70)
        print("✅ Demo complete!")
        print("="*70)
    else:
        print("\n❌ No ball detected in video")
    

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("\n📝 Usage:")
        print("   python demo_manual_selection.py <video_file>")
        print("\n📝 Example:")
        print("   python demo_manual_selection.py uploads/cricket.mp4")
        print("\n💡 Tip: Place your cricket video in the 'uploads' folder")
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"\n❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    demo_manual_selection(video_path)
