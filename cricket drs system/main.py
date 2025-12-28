"""
Cricket DRS System - FastAPI Application

Refactored to use modular architecture.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from drs.tracker import BallTracker, VideoAnnotator
from drs.trajectory import TrajectoryAnalyzer
from drs.decision import DecisionMaker
from drs.config import (
    BallDetectionConfig, 
    TrajectoryConfig, 
    DecisionConfig,
    StumpRegion,
    VideoConfig,
    create_stump_region_for_frame
)
from drs.detectors import ColorBasedDetector, MultiColorDetector, YOLODetector, AdaptiveBallDetector
from drs.video_io import VideoProcessor


# Initialize FastAPI app
app = FastAPI(
    title="Cricket DRS System",
    description="Decision Review System for cricket using video analysis",
    version="1.0.0"
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Load configurations
VIDEO_CONFIG = VideoConfig()
DETECTION_CONFIG = BallDetectionConfig()
TRAJECTORY_CONFIG = TrajectoryConfig()
DECISION_CONFIG = DecisionConfig()

# Default stump region configuration (pixels - needs calibration based on video)
DEFAULT_STUMP_REGION = StumpRegion(
    x=640,
    y_top=300,
    y_bottom=500
)


def validate_video_file(file: UploadFile) -> None:
    """
    Validate uploaded video file
    
    Args:
        file: Uploaded file object
        
    Raises:
        HTTPException: If file is invalid
    """
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in VIDEO_CONFIG.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(VIDEO_CONFIG.allowed_extensions)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cricket DRS System API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Upload and analyze cricket video",
            "GET /download/{filename}": "Download processed video"
        }
    }


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    stump_x: Optional[int] = None,
    stump_y_top: Optional[int] = None,
    stump_y_bottom: Optional[int] = None,
    manual_selection: Optional[bool] = True  # NEW: Enable manual ball selection by default
):
    """
    Analyze cricket video for DRS decision
    
    Args:
        file: Video file (.mp4, .avi, .mov)
        stump_x: X coordinate of stumps (optional, uses default if not provided)
        stump_y_top: Top Y coordinate of stumps (optional)
        stump_y_bottom: Bottom Y coordinate of stumps (optional)
        manual_selection: Enable manual ball selection (default: True)
        
    Returns:
        JSON response with:
            - decision: OUT/NOT OUT/INCONCLUSIVE
            - confidence: Confidence score
            - impact_point: Predicted impact coordinates
            - output_video: Path to annotated video
            - tracking_stats: Tracking quality metrics
    """
    # Validate file
    validate_video_file(file)
    
    # Generate unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    input_filename = f"{unique_id}_input{file_ext}"
    output_filename = f"{unique_id}_output.mp4"
    
    input_path = UPLOAD_DIR / input_filename
    output_path = OUTPUT_DIR / output_filename
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get video dimensions for auto-scaling stump region
        import cv2
        cap = cv2.VideoCapture(str(input_path))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Configure stump region (auto-scale based on frame size if not provided)
        if stump_x is None and stump_y_top is None and stump_y_bottom is None:
            # Auto-generate stump region based on frame size
            stump_region = create_stump_region_for_frame(frame_width, frame_height)
            print(f"✓ Auto-configured stumps: x={stump_region.x:.0f}, y=({stump_region.y_top:.0f}, {stump_region.y_bottom:.0f})")
        else:
            # Use provided values or defaults
            stump_region = StumpRegion(
                x=stump_x if stump_x is not None else DEFAULT_STUMP_REGION.x,
                y_top=stump_y_top if stump_y_top is not None else DEFAULT_STUMP_REGION.y_top,
                y_bottom=stump_y_bottom if stump_y_bottom is not None else DEFAULT_STUMP_REGION.y_bottom
            )
        
        # Initialize components with configurations
        # Use AdaptiveBallDetector (learns from manual selection!)
        detector = AdaptiveBallDetector() if manual_selection else MultiColorDetector()
        if manual_selection:
            print("✓ Using AdaptiveBallDetector (learns ball color from your selection)")
        else:
            print("✓ Using MultiColorDetector (color-based cricket ball detection)")
        
        # Enable manual selection for accurate tracking
        tracker = BallTracker(
            detector=detector, 
            config=DETECTION_CONFIG, 
            use_kalman=True,
            use_manual_selection=manual_selection
        )
        if manual_selection:
            print("🎯 Manual ball selection enabled - you'll click on the ball")
        print("✓ Kalman filter enabled for smooth tracking")
        analyzer = TrajectoryAnalyzer(config=TRAJECTORY_CONFIG)
        decision_maker = DecisionMaker(config=DECISION_CONFIG)
        annotator = VideoAnnotator()
        
        # Step 1: Track ball in video (process EVERY frame for fast balls)
        print("⚡ Processing every frame for fast ball tracking...")
        tracking_result = tracker.track_video(str(input_path), frame_skip=1)
        
        if tracking_result['frames_tracked'] == 0:
            raise HTTPException(
                status_code=422,
                detail="No ball detected in video. Please ensure video contains a visible red cricket ball."
            )
        
        # Step 2: Analyze trajectory
        trajectory_analysis = analyzer.analyze_trajectory(
            tracking_result['trajectory'],
            stump_region
        )
        
        # Step 3: Make decision
        decision_result = decision_maker.make_decision(
            trajectory_analysis,
            tracking_result
        )
        
        # Step 4: Draw trajectory on video
        annotator.annotate_video(
            str(input_path),
            str(output_path),
            tracking_result['trajectory']
        )
        
        # Format response
        response = decision_maker.format_response(decision_result)
        response['output_video'] = f"/download/{output_filename}"
        response['ball_color'] = detector.get_last_detected_color()  # Add detected ball color
        response['video_info'] = {
            'fps': tracking_result['fps'],
            'total_frames': tracking_result['total_frames'],
            'frame_size': tracking_result['frame_size']
        }
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    finally:
        # Clean up uploaded file
        if input_path.exists():
            input_path.unlink()


@app.get("/download/{filename}")
async def download_video(filename: str):
    """
    Download processed video file
    
    Args:
        filename: Name of the output file
        
    Returns:
        Video file
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )


@app.delete("/cleanup")
async def cleanup_files():
    """
    Clean up all uploaded and output files
    
    Returns:
        Cleanup status
    """
    try:
        # Remove all files from uploads and outputs
        for folder in [UPLOAD_DIR, OUTPUT_DIR]:
            for file_path in folder.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
        
        return {
            "status": "success",
            "message": "All temporary files cleaned up"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during cleanup: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
