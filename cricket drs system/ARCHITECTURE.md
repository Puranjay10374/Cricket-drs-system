# Cricket DRS System - Modular Architecture

## Overview

The Cricket DRS System has been refactored into a highly modular architecture that separates concerns and makes the codebase more maintainable, testable, and extensible.

## Architecture Diagram

```
cricket-drs-system/
├── main.py                    # FastAPI application (orchestration layer)
├── drs/                       # Core package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration dataclasses
│   ├── detectors.py          # Ball detection strategies
│   ├── tracker.py            # Ball tracking coordination
│   ├── trajectory.py         # Trajectory prediction
│   ├── decision.py           # Decision making logic
│   ├── video_io.py           # Video reading/writing
│   └── renderer.py           # Visualization and overlays
├── examples.py               # Usage examples
└── ARCHITECTURE.md           # This file
```

## Module Breakdown

### 1. **config.py** - Configuration Management

**Purpose**: Centralize all configuration parameters using dataclasses.

**Classes**:
- `BallDetectionConfig` - Detection parameters (colors, sizes, morphology)
- `VideoConfig` - Video processing settings (frame skip, file limits)
- `TrajectoryConfig` - Polynomial fitting parameters
- `StumpRegion` - Stump position and validation
- `DecisionConfig` - Decision thresholds and weights
- `RenderConfig` - Rendering colors and styles

**Benefits**:
- Type-safe configuration
- Easy to modify parameters
- No magic numbers in code
- Supports multiple configurations

**Example**:
```python
config = BallDetectionConfig(
    min_area=100,
    max_area=3000,
    color_lower1=(0, 150, 100)
)
```

---

### 2. **detectors.py** - Detection Strategies

**Purpose**: Implement different ball detection algorithms using the Strategy pattern.

**Classes**:
- `BallDetector` (ABC) - Abstract interface for all detectors
- `ColorBasedDetector` - HSV color-based detection
- `ContourBasedDetector` - Shape circularity detection
- `HybridDetector` - Combines multiple strategies

**Benefits**:
- Easy to add new detection algorithms
- Swap strategies at runtime
- Test different approaches
- Polymorphic design

**Example**:
```python
# Use color-based detection
detector = ColorBasedDetector(config)

# Or switch to contour-based
detector = ContourBasedDetector(config)

# Pass to tracker
tracker = BallTracker(detector=detector)
```

---

### 3. **video_io.py** - Video I/O Operations

**Purpose**: Abstract video reading and writing operations.

**Classes**:
- `VideoReader` - Context manager for reading videos
- `VideoWriter` - Context manager for writing videos
- `VideoProcessor` - High-level utilities (validation, info)

**Benefits**:
- Clean resource management (context managers)
- Separation from business logic
- Reusable across different parts of the system
- Proper error handling

**Example**:
```python
with VideoReader(input_path) as reader:
    props = reader.get_properties()
    
    with VideoWriter(output_path, props['fps'], props['frame_size']) as writer:
        for frame_num, frame in reader.frames():
            # Process frame
            writer.write_frame(processed_frame)
```

---

### 4. **renderer.py** - Visualization

**Purpose**: Handle all drawing and overlay operations.

**Classes**:
- `TrajectoryRenderer` - Draws trajectory overlays on frames

**Methods**:
- `draw_ball()` - Ball marker
- `draw_trajectory_path()` - Tracked path
- `draw_predicted_path()` - Predicted trajectory
- `draw_stump_region()` - Stump indicator
- `draw_impact_point()` - Impact marker
- `draw_decision_overlay()` - Decision text

**Benefits**:
- Separates visualization from detection
- Configurable colors and styles
- Reusable rendering components
- Easy to add new visualizations

**Example**:
```python
renderer = TrajectoryRenderer(render_config)
frame = renderer.draw_ball(frame, x, y, radius)
frame = renderer.draw_trajectory_path(frame, points)
```

---

### 5. **tracker.py** - Ball Tracking (Refactored)

**Purpose**: Coordinate ball tracking using pluggable detectors.

**Changes from original**:
- ❌ Removed: Hardcoded detection logic
- ✅ Added: Dependency injection for detectors
- ✅ Added: Uses VideoReader for I/O
- ✅ Simplified: Delegates rendering to TrajectoryRenderer

**Class**:
- `BallTracker` - Tracks ball using injected detector

**Example**:
```python
detector = ColorBasedDetector(config)
tracker = BallTracker(detector=detector, config=config)
result = tracker.track_video(video_path)
```

---

### 6. **trajectory.py** - Trajectory Prediction (Refactored)

**Purpose**: Predict ball trajectory using polynomial fitting.

**Changes from original**:
- ✅ Added: `PolynomialFitter` - Separated math operations
- ✅ Added: Configuration-based initialization
- ✅ Changed: Uses `StumpRegion` objects instead of dicts

**Classes**:
- `PolynomialFitter` - Pure mathematical operations
- `TrajectoryPredictor` - Business logic and orchestration

**Example**:
```python
config = TrajectoryConfig(polynomial_degree=2)
predictor = TrajectoryPredictor(config=config)

stump_region = StumpRegion(x=640, y_top=300, y_bottom=500)
analysis = predictor.analyze_trajectory(trajectory, stump_region)
```

---

### 7. **decision.py** - Decision Making (Refactored)

**Purpose**: Make OUT/NOT OUT decisions based on trajectory.

**Changes from original**:
- ✅ Added: `ConfidenceCalculator` - Separated confidence logic
- ✅ Added: Configuration-based initialization
- ✅ Simplified: Uses config for thresholds

**Classes**:
- `ConfidenceCalculator` - Calculates confidence scores
- `DRSDecision` - Makes final decisions

**Example**:
```python
config = DecisionConfig(confidence_threshold=0.7)
decision_maker = DRSDecision(config=config)
decision = decision_maker.make_decision(trajectory_analysis, tracking_info)
```

---

### 8. **main.py** - FastAPI Application (Refactored)

**Purpose**: HTTP API orchestration layer.

**Changes from original**:
- ✅ Uses modular components
- ✅ Configuration-driven
- ✅ Cleaner dependency management

**Example**:
```python
# Initialize with configs
detector = ColorBasedDetector(DETECTION_CONFIG)
tracker = BallTracker(detector=detector, config=DETECTION_CONFIG)
predictor = TrajectoryPredictor(config=TRAJECTORY_CONFIG)
decision_maker = DRSDecision(config=DECISION_CONFIG)

# Use in pipeline
tracking_result = tracker.track_video(video_path)
trajectory_analysis = predictor.analyze_trajectory(...)
decision = decision_maker.make_decision(...)
```

---

## Design Patterns Used

### 1. **Strategy Pattern**
- **Where**: `detectors.py`
- **Why**: Multiple ball detection algorithms
- **Benefit**: Easy to add new detection methods

### 2. **Dependency Injection**
- **Where**: `tracker.py`, `trajectory.py`, `decision.py`
- **Why**: Loose coupling between components
- **Benefit**: Testable, configurable, extensible

### 3. **Context Manager**
- **Where**: `video_io.py`
- **Why**: Proper resource cleanup
- **Benefit**: Prevents resource leaks

### 4. **Facade Pattern**
- **Where**: `VideoProcessor`
- **Why**: Simplified interface for complex operations
- **Benefit**: Easy to use high-level API

### 5. **Data Transfer Object (DTO)**
- **Where**: `config.py` dataclasses
- **Why**: Type-safe configuration passing
- **Benefit**: Clear contracts between components

---

## Modularity Benefits

### Before Refactoring:
```python
# All in one class - hard to test, configure, or extend
class BallTracker:
    def __init__(self, min_area, max_area, color1, color2, ...):
        self.min_area = min_area
        # ... 10+ parameters
        
    def detect_ball(self, frame):
        # 50 lines of detection logic
        
    def track_video(self, path):
        # Mixed concerns: I/O + detection + tracking
        
    def draw_trajectory(self, ...):
        # Rendering mixed with tracking
```

### After Refactoring:
```python
# Clean separation
config = BallDetectionConfig()  # Type-safe config
detector = ColorBasedDetector(config)  # Pluggable strategy
tracker = BallTracker(detector=detector)  # Focused responsibility

# Easy to test
def test_color_detector():
    frame = load_test_frame()
    detector = ColorBasedDetector(config)
    result = detector.detect(frame)
    assert result is not None

# Easy to extend
class MLBasedDetector(BallDetector):
    def detect(self, frame):
        return self.model.predict(frame)
```

---

## Testing Strategy

### Unit Tests (Easy with new architecture)

```python
# Test detector independently
def test_color_detector():
    config = BallDetectionConfig()
    detector = ColorBasedDetector(config)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result is None  # No ball in blank frame

# Test polynomial fitter independently
def test_polynomial_fitter():
    fitter = PolynomialFitter(degree=2)
    points = [(0, 0), (1, 1), (2, 4)]  # y = x^2
    result = fitter.fit(points)
    assert result['fit_quality'] > 0.99

# Test confidence calculator independently
def test_confidence_calculator():
    config = DecisionConfig()
    calc = ConfidenceCalculator(config)
    confidence = calc.calculate(0.8, 0.9, 20)
    assert 0 <= confidence <= 1
```

### Integration Tests

```python
def test_full_pipeline():
    # Create components
    tracker = BallTracker()
    predictor = TrajectoryPredictor()
    decision_maker = DRSDecision()
    
    # Process test video
    result = tracker.track_video("test_video.mp4")
    analysis = predictor.analyze_trajectory(result['trajectory'], stump_region)
    decision = decision_maker.make_decision(analysis, result)
    
    assert decision['decision'] in ['OUT', 'NOT OUT', 'INCONCLUSIVE']
```

---

## Extension Points

### Adding a New Detector

```python
# 1. Create new detector class
class TemplateMatchingDetector(BallDetector):
    def __init__(self, template_path: str):
        self.template = cv2.imread(template_path)
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        result = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        # ... implementation
        return (x, y, radius)

# 2. Use it
detector = TemplateMatchingDetector("ball_template.png")
tracker = BallTracker(detector=detector)
```

### Adding Custom Renderer

```python
# Extend renderer with custom overlays
class CustomRenderer(TrajectoryRenderer):
    def draw_spin_indicator(self, frame, spin_rate):
        # Custom visualization
        pass
```

### Adding New Decision Criteria

```python
# Extend decision maker
class AdvancedDRSDecision(DRSDecision):
    def make_decision(self, trajectory, tracking):
        # Call parent
        decision = super().make_decision(trajectory, tracking)
        
        # Add custom logic (e.g., spin analysis)
        if self.analyze_spin(trajectory):
            decision['spin_detected'] = True
        
        return decision
```

---

## Migration Guide

### Old Code:
```python
tracker = BallTracker(
    min_area=50,
    max_area=5000,
    color_lower1=(0, 120, 70),
    # ... many parameters
)
```

### New Code:
```python
config = BallDetectionConfig(
    min_area=50,
    max_area=5000,
    color_lower1=(0, 120, 70)
)
detector = ColorBasedDetector(config)
tracker = BallTracker(detector=detector, config=config)
```

---

## Performance Considerations

### No Performance Penalty
- Abstraction is at class level, not per-frame
- Context managers use same underlying OpenCV calls
- Configuration objects have zero runtime overhead (dataclasses)
- Strategy pattern uses direct method calls (no reflection)

### Performance Gains
- `VideoReader` iterator is more memory efficient
- Frame skipping is cleaner and more controllable
- Easier to profile individual components

---

## Future Enhancements

The modular architecture makes these additions easier:

1. **Machine Learning Detectors**
   ```python
   class YOLODetector(BallDetector):
       def __init__(self, model_path):
           self.model = YOLO(model_path)
   ```

2. **3D Trajectory Analysis**
   ```python
   class Trajectory3DPredictor(TrajectoryPredictor):
       def analyze_trajectory_3d(self, multi_camera_data):
           pass
   ```

3. **Real-time Processing**
   ```python
   class RealtimeVideoReader(VideoReader):
       def __init__(self, camera_index):
           self.cap = cv2.VideoCapture(camera_index)
   ```

4. **Custom Metrics**
   ```python
   class SpinAnalyzer:
       def analyze_spin(self, trajectory):
           pass
   ```

---

## Summary

The refactored architecture provides:

✅ **Modularity** - Each module has a single, well-defined purpose  
✅ **Testability** - Components can be tested in isolation  
✅ **Extensibility** - Easy to add new features without modifying existing code  
✅ **Configurability** - Type-safe configuration management  
✅ **Maintainability** - Clear separation of concerns  
✅ **Reusability** - Components can be used in different contexts  
✅ **Flexibility** - Swap implementations at runtime  

The system follows SOLID principles and uses established design patterns to create a professional, production-ready codebase.
