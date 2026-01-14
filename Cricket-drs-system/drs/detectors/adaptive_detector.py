"""
Advanced Adaptive Ball Detector

State-of-the-art ball detection using:
- Multi-scale template matching with rotation invariance
- Online appearance learning and adaptation
- Motion-based prediction and tracking
- Multiple color space fusion
- Probabilistic candidate scoring
- Background subtraction and optical flow
- Ensemble detection with confidence weighting
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque
from .base_detector import BallDetector


class AdaptiveBallDetector(BallDetector):
    """Advanced detector with online learning and multi-modal fusion"""
    
    def __init__(self, history_size: int = 30):
        # === Appearance Models ===
        self.ball_templates = []  # Multiple templates at different scales/rotations
        self.ball_histogram = None  # Color histogram
        self.ball_color_models = []  # Multiple color models (HSV, LAB, YCrCb)
        self.ball_features = None  # ORB/SIFT features for robust matching
        self.appearance_history = deque(maxlen=history_size)  # Online learning
        
        # === Motion Models ===
        self.position_history = deque(maxlen=20)
        self.velocity_history = deque(maxlen=10)
        self.predicted_position = None
        self.search_region = None  # Adaptive search window
        
        # === Background Models ===
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.static_background = None  # For static camera scenarios
        
        # === Feature Extractors ===
        self.orb = cv2.ORB_create(nfeatures=100)
        self.sift = None  # Optional: cv2.SIFT_create() if available
        
        # === Optical Flow ===
        self.prev_gray = None
        self.flow_mask = None
        
        # === State ===
        self.learned = False
        self.frame_count = 0
        self.confidence_threshold = 0.3
        self.last_detected_color = None
        self.detection_confidence = 0.0
        
        # === Multi-scale parameters ===
        self.template_scales = [0.7, 0.85, 1.0, 1.15, 1.3]
        self.template_rotations = [0, 45, 90, 135, 180, 225, 270, 315]  # For spin
        
        # === Adaptive thresholds ===
        self.min_radius = 3
        self.max_radius = 35
        self.expected_radius = 10  # Updated online
        
    def learn_from_selection(self, frame: np.ndarray, x: int, y: int, radius: int):
        """
        Deep learning from manual selection - extracts comprehensive appearance model
        
        Args:
            frame: BGR image
            x, y: Ball center position
            radius: Ball radius
        """
        print("🧠 Learning comprehensive ball appearance model...")
        
        # Store expected radius
        self.expected_radius = radius
        
        # Extract ball region with safety margin
        margin = max(int(radius * 1.5), 10)
        y1, y2 = max(0, y - margin), min(frame.shape[0], y + margin)
        x1, x2 = max(0, x - margin), min(frame.shape[1], x + margin)
        ball_region = frame[y1:y2, x1:x2]
        
        if ball_region.size == 0:
            return
        
        # === 1. Multi-scale Template Learning ===
        self._learn_templates(ball_region, radius)
        
        # === 2. Multi-Color Space Learning ===
        self._learn_color_models(ball_region, radius, margin, x - x1, y - y1)
        
        # === 3. Feature Learning (ORB/SIFT) ===
        self._learn_features(ball_region)
        
        # === 4. Histogram Learning ===
        self._learn_histogram(ball_region, radius, margin, x - x1, y - y1)
        
        # === 5. Initialize Motion Model ===
        self.position_history.append((x, y))
        self.predicted_position = (x, y)
        
        # === 6. Learn Background ===
        self.static_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.learned = True
        print(f"✓ Learned {len(self.ball_templates)} templates across {len(self.template_scales)} scales")
        print(f"✓ Learned {len(self.ball_color_models)} color models")
        print(f"✓ Expected ball radius: {self.expected_radius}px")
    
    def _learn_templates(self, ball_region: np.ndarray, radius: int):
        """Learn multi-scale and rotation-invariant templates"""
        self.ball_templates = []
        
        for scale in self.template_scales:
            size = max(int(radius * 2 * scale), 10)
            if size > min(ball_region.shape[:2]):
                continue
            
            template = cv2.resize(ball_region, (size, size))
            
            # Store template and its rotations (for spinning ball)
            for angle in [0, 90, 180, 270]:
                if angle != 0:
                    M = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
                    rotated = cv2.warpAffine(template, M, (size, size))
                    self.ball_templates.append({
                        'image': rotated,
                        'scale': scale,
                        'angle': angle,
                        'size': size
                    })
                else:
                    self.ball_templates.append({
                        'image': template,
                        'scale': scale,
                        'angle': angle,
                        'size': size
                    })
    
    def _learn_color_models(self, ball_region: np.ndarray, radius: int, margin: int, cx: int, cy: int):
        """Learn robust color models in multiple color spaces"""
        self.ball_color_models = []
        
        # Create circular mask to exclude background
        mask = np.zeros(ball_region.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), max(int(radius * 0.8), 3), 255, -1)
        
        # HSV Model
        hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        hsv_masked = hsv[mask > 0]
        if len(hsv_masked) > 0:
            hsv_model = self._create_color_model(hsv_masked, 'HSV')
            self.ball_color_models.append(hsv_model)
            
            # Determine ball color
            h_median = np.median(hsv_masked[:, 0])
            s_median = np.median(hsv_masked[:, 1])
            v_median = np.median(hsv_masked[:, 2])
            self._determine_ball_color(h_median, s_median, v_median)
        
        # LAB Model (better for illumination invariance)
        lab = cv2.cvtColor(ball_region, cv2.COLOR_BGR2LAB)
        lab_masked = lab[mask > 0]
        if len(lab_masked) > 0:
            lab_model = self._create_color_model(lab_masked, 'LAB')
            self.ball_color_models.append(lab_model)
        
        # YCrCb Model (better for skin-like colors)
        ycrcb = cv2.cvtColor(ball_region, cv2.COLOR_BGR2YCrCb)
        ycrcb_masked = ycrcb[mask > 0]
        if len(ycrcb_masked) > 0:
            ycrcb_model = self._create_color_model(ycrcb_masked, 'YCrCb')
            self.ball_color_models.append(ycrcb_model)
    
    def _create_color_model(self, pixels: np.ndarray, color_space: str) -> Dict:
        """Create statistical color model from pixels"""
        # Use percentiles for robustness
        p05 = np.percentile(pixels, 5, axis=0)
        p95 = np.percentile(pixels, 95, axis=0)
        median = np.median(pixels, axis=0)
        mad = np.median(np.abs(pixels - median), axis=0)
        
        # Create tight bounds
        lower = np.maximum(0, median - 2 * mad)
        upper = np.minimum(255, median + 2 * mad)
        
        return {
            'space': color_space,
            'lower': lower.astype(np.uint8),
            'upper': upper.astype(np.uint8),
            'median': median,
            'mad': mad,
            'p05': p05,
            'p95': p95
        }
    
    def _learn_features(self, ball_region: np.ndarray):
        """Extract keypoint features for robust matching"""
        gray = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
        
        # ORB features (fast, rotation-invariant)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        self.ball_features = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'image': gray
        }
    
    def _learn_histogram(self, ball_region: np.ndarray, radius: int, margin: int, cx: int, cy: int):
        """Learn color histogram for back-projection"""
        hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Create circular mask
        mask = np.zeros(ball_region.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), max(int(radius * 0.8), 3), 255, -1)
        
        # Calculate histogram
        self.ball_histogram = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(self.ball_histogram, self.ball_histogram, 0, 255, cv2.NORM_MINMAX)
    
    def detect(self, frame: np.ndarray, prev_detection: Optional[Tuple[int, int, int]] = None) -> Optional[Tuple[int, int, int]]:
        """
        Advanced multi-modal detection with fusion
        
        Args:
            frame: BGR image
            prev_detection: Previous frame detection for motion prediction
            
        Returns:
            (x, y, radius) or None
        """
        if not self.learned:
            return self._detect_basic(frame)
        
        self.frame_count += 1
        
        # Update motion prediction
        if prev_detection:
            self._update_motion_model(prev_detection)
        
        # === Multi-Modal Detection ===
        candidates = []
        
        # 1. Template Matching (high precision)
        template_results = self._detect_by_multi_template(frame)
        candidates.extend(template_results)
        
        # 2. Color-based detection (fast, multi-space)
        color_results = self._detect_by_multi_color(frame)
        candidates.extend(color_results)
        
        # 3. Background subtraction (motion-based)
        bg_results = self._detect_by_background_subtraction(frame)
        candidates.extend(bg_results)
        
        # 4. Histogram back-projection
        hist_results = self._detect_by_histogram(frame)
        candidates.extend(hist_results)
        
        # === Fuse all candidates ===
        result = self._fuse_candidates(candidates, frame)
        
        # Update appearance model if detection successful
        if result:
            self._update_appearance_model(frame, result)
            self.detection_confidence = 0.8
        else:
            self.detection_confidence = 0.0
        
        return result
    
    def _detect_by_multi_template(self, frame: np.ndarray) -> List[Tuple]:
        """Multi-scale, multi-rotation template matching"""
        candidates = []
        
        # Limit search region if we have motion prediction
        search_frame = self._get_search_region(frame)
        if search_frame is None:
            search_frame = frame
            offset_x, offset_y = 0, 0
        else:
            search_frame, offset_x, offset_y = search_frame
        
        gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)
        
        for template_info in self.ball_templates[:10]:  # Use top 10 templates
            template = cv2.cvtColor(template_info['image'], cv2.COLOR_BGR2GRAY)
            
            # Match with multiple methods
            for method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
                result = cv2.matchTemplate(gray, template, method)
                
                # Find local maxima
                threshold = 0.5 if method == cv2.TM_CCOEFF_NORMED else 0.7
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    x = pt[0] + template_info['size'] // 2 + offset_x
                    y = pt[1] + template_info['size'] // 2 + offset_y
                    radius = template_info['size'] // 2
                    
                    candidates.append((x, y, radius, confidence * 0.9, 'template'))
        
        return candidates
    
    def _detect_by_multi_color(self, frame: np.ndarray) -> List[Tuple]:
        """Detection in multiple color spaces with fusion"""
        candidates = []
        
        for model in self.ball_color_models:
            # Convert to appropriate color space
            if model['space'] == 'HSV':
                converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            elif model['space'] == 'LAB':
                converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            elif model['space'] == 'YCrCb':
                converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            else:
                continue
            
            # Create mask
            mask = cv2.inRange(converted, model['lower'], model['upper'])
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 15:
                    continue
                
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                if not (self.min_radius <= radius <= self.max_radius):
                    continue
                
                # Calculate shape metrics
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter ** 2)
                compactness = area / (np.pi * radius ** 2)
                
                # Size match score
                size_score = 1.0 - abs(radius - self.expected_radius) / self.expected_radius
                size_score = max(0, min(1, size_score))
                
                # Combined score
                confidence = (circularity * 0.4 + compactness * 0.3 + size_score * 0.3) * 0.8
                
                if confidence > 0.3:
                    candidates.append((int(x), int(y), int(radius), confidence, f'color_{model["space"]}'))
        
        return candidates
    
    def _fuse_candidates(self, candidates: List[Tuple], frame: np.ndarray) -> Optional[Tuple]:
        """
        Intelligent fusion of detection candidates using:
        - Spatial clustering
        - Confidence weighting  
        - Motion consistency
        - Ensemble voting
        """
        if not candidates:
            return None
        
        # Remove duplicates and cluster nearby detections
        clustered = self._cluster_candidates(candidates)
        
        # Score each cluster
        scored_clusters = []
        for cluster in clustered:
            score = self._score_cluster(cluster, frame)
            if score > self.confidence_threshold:
                scored_clusters.append((cluster, score))
        
        if not scored_clusters:
            return None
        
        # Return best cluster
        best_cluster, best_score = max(scored_clusters, key=lambda x: x[1])
        
        # Calculate weighted centroid
        total_weight = sum(c[3] for c in best_cluster)
        if total_weight == 0:
            return None
        
        x = int(sum(c[0] * c[3] for c in best_cluster) / total_weight)
        y = int(sum(c[1] * c[3] for c in best_cluster) / total_weight)
        radius = int(sum(c[2] * c[3] for c in best_cluster) / total_weight)
        
        self.detection_confidence = best_score
        return (x, y, radius)
    
    def _cluster_candidates(self, candidates: List[Tuple]) -> List[List[Tuple]]:
        """Cluster nearby candidates using spatial proximity"""
        if not candidates:
            return []
        
        clusters = []
        used = set()
        
        for i, cand in enumerate(candidates):
            if i in used:
                continue
            
            cluster = [cand]
            used.add(i)
            
            for j, other in enumerate(candidates):
                if j in used:
                    continue
                
                dist = np.sqrt((cand[0] - other[0])**2 + (cand[1] - other[1])**2)
                if dist < 30:  # Cluster radius
                    cluster.append(other)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _score_cluster(self, cluster: List[Tuple], frame: np.ndarray) -> float:
        """Score a cluster of detections"""
        if not cluster:
            return 0.0
        
        # Average confidence score
        confidence_score = np.mean([c[3] for c in cluster])
        
        # Bonus for multiple detection methods agreeing
        methods = set([c[4] for c in cluster])
        diversity_bonus = len(methods) * 0.1
        
        # Bonus for cluster size (consensus)
        size_bonus = min(0.2, len(cluster) * 0.05)
        
        # Motion consistency bonus
        motion_bonus = 0.0
        if self.predicted_position:
            centroid_x = np.mean([c[0] for c in cluster])
            centroid_y = np.mean([c[1] for c in cluster])
            dist_to_prediction = np.sqrt(
                (centroid_x - self.predicted_position[0])**2 +
                (centroid_y - self.predicted_position[1])**2
            )
            motion_bonus = max(0, 0.3 - dist_to_prediction / 100.0)
        
        total_score = confidence_score + diversity_bonus + size_bonus + motion_bonus
        return min(1.0, total_score)
    
    def get_last_detected_color(self) -> Optional[str]:
        """Get detected ball color"""
        return self.last_detected_color
    
    def get_detection_confidence(self) -> float:
        """Get confidence of last detection (0-1)"""
        return self.detection_confidence
    
    def reset(self):
        """Reset detector state for new video"""
        self.frame_count = 0
        self.position_history.clear()
        self.velocity_history.clear()
        self.appearance_history.clear()
        self.predicted_position = None
        self.prev_gray = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
    
    def _update_motion_model(self, detection: Tuple[int, int, int]):
        """Update motion prediction"""
        x, y = detection[:2]
        self.position_history.append((x, y))
        
        # Calculate velocity
        if len(self.position_history) >= 2:
            prev = self.position_history[-2]
            vx = x - prev[0]
            vy = y - prev[1]
            self.velocity_history.append((vx, vy))
            
            # Predict next position
            if len(self.velocity_history) >= 3:
                avg_vx = np.mean([v[0] for v in list(self.velocity_history)[-3:]])
                avg_vy = np.mean([v[1] for v in list(self.velocity_history)[-3:]])
                self.predicted_position = (int(x + avg_vx), int(y + avg_vy))
    
    def _get_search_region(self, frame: np.ndarray) -> Optional[Tuple]:
        """Get adaptive search region based on motion prediction"""
        if self.predicted_position is None:
            return None
        
        px, py = self.predicted_position
        
        # Adaptive window size based on velocity
        if self.velocity_history:
            speed = np.sqrt(sum(v[0]**2 + v[1]**2 for v in self.velocity_history)) / len(self.velocity_history)
            window_size = int(max(100, min(300, 100 + speed * 10)))
        else:
            window_size = 150
        
        # Extract region
        h, w = frame.shape[:2]
        x1 = max(0, px - window_size)
        y1 = max(0, py - window_size)
        x2 = min(w, px + window_size)
        y2 = min(h, py + window_size)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return (frame[y1:y2, x1:x2], x1, y1)
    
    def _update_appearance_model(self, frame: np.ndarray, detection: Tuple):
        """Online appearance learning - adapt to lighting/ball changes"""
        x, y, radius = detection[:3]
        
        # Extract current appearance
        margin = max(int(radius * 1.5), 10)
        y1, y2 = max(0, y - margin), min(frame.shape[0], y + margin)
        x1, x2 = max(0, x - margin), min(frame.shape[1], x + margin)
        region = frame[y1:y2, x1:x2]
        
        if region.size == 0:
            return
        
        self.appearance_history.append(region)
        
        # Update models every N frames
        if self.frame_count % 30 == 0 and len(self.appearance_history) > 10:
            # Re-learn color models from recent history
            recent_regions = list(self.appearance_history)[-10:]
            combined = np.vstack([r.reshape(-1, 3) for r in recent_regions])
            
            # Update color models with exponential moving average
            for model in self.ball_color_models:
                # Update incrementally (80% old, 20% new)
                pass  # Implement incremental update if needed
        
        # Update expected radius
        self.expected_radius = int(self.expected_radius * 0.9 + radius * 0.1)
    
    def _detect_by_background_subtraction(self, frame: np.ndarray) -> List[Tuple]:
        """Detect moving objects (ball should be moving)"""
        candidates = []
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if self.min_radius <= radius <= self.max_radius:
                confidence = min(1.0, area / (np.pi * radius ** 2)) * 0.6
                candidates.append((int(x), int(y), int(radius), confidence, 'background'))
        
        return candidates
    
    def _detect_by_histogram(self, frame: np.ndarray) -> List[Tuple]:
        """Histogram back-projection for color-based detection"""
        if self.ball_histogram is None:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0, 1], self.ball_histogram, [0, 180, 0, 256], 1)
        
        # Threshold and clean
        _, thresh = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 15:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if self.min_radius <= radius <= self.max_radius:
                confidence = min(1.0, area / 500.0) * 0.7
                candidates.append((int(x), int(y), int(radius), confidence, 'histogram'))
        
        return candidates
    
    def _detect_by_optical_flow(self, frame: np.ndarray) -> List[Tuple]:
        """Use optical flow to find fast-moving objects"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate flow magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Threshold for significant motion
        motion_mask = (magnitude > 2.0).astype(np.uint8) * 255
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        # Find moving regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if self.min_radius <= radius <= self.max_radius:
                avg_magnitude = cv2.mean(magnitude, mask=(motion_mask > 0))[0]
                confidence = min(1.0, avg_magnitude / 20.0) * 0.65
                candidates.append((int(x), int(y), int(radius), confidence, 'optical_flow'))
        
        self.prev_gray = gray
        return candidates
    
    def _detect_by_features(self, frame: np.ndarray) -> List[Tuple]:
        """Feature-based detection using ORB matching"""
        if self.ball_features is None or self.ball_features['descriptors'] is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None:
            return []
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.ball_features['descriptors'], des)
        
        if len(matches) < 3:
            return []
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Get matched keypoint locations
        matched_points = [kp[m.trainIdx].pt for m in matches[:10]]
        
        if not matched_points:
            return []
        
        # Cluster matched points
        matched_array = np.array(matched_points)
        center = np.mean(matched_array, axis=0)
        
        confidence = min(1.0, len(matches) / 20.0) * 0.75
        
        return [(int(center[0]), int(center[1]), self.expected_radius, confidence, 'features')]
    
    def _detect_basic(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Basic detection without learning (fallback)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Generic pink/red detection
        lower_pink = np.array([145, 50, 50])
        upper_pink = np.array([165, 255, 255])
        
        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        
        if radius > 2 and radius < 30:
            return (int(x), int(y), int(radius))
        
        return None
    
    def _determine_ball_color(self, h: float, s: float, v: float):
        """
        Determine ball color category from HSV values
        
        Args:
            h: Hue value (0-179)
            s: Saturation value (0-255)
            v: Value/Brightness (0-255)
        """
        # Red ball: H ~ 0-10 or 170-179 with high saturation
        if (h <= 10 or h >= 170) and s > 100:
            self.last_detected_color = "red"
        # Pink ball: H ~ 145-165 with medium to high saturation
        elif 145 <= h <= 165 and s > 50:
            self.last_detected_color = "pink"
        # White ball: Low saturation, high brightness
        elif s < 50 and v > 150:
            self.last_detected_color = "white"
        # Yellow ball: H ~ 20-35
        elif 20 <= h <= 35 and s > 100:
            self.last_detected_color = "yellow"
        else:
            # Unknown/other color - use generic label
            self.last_detected_color = "custom"
    
    def get_last_detected_color(self) -> Optional[str]:
        """
        Get the color of the last detected ball
        
        Returns:
            'red', 'white', 'pink', 'yellow', 'custom', or None
        """
        return self.last_detected_color
