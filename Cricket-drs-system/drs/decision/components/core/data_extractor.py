"""
Data Extraction Module

Extracts and structures data from tracking and trajectory analysis results.
"""

from typing import Dict, Tuple, Optional


class DataExtractor:
    """Extracts structured data from analysis results"""
    
    @staticmethod
    def extract_intersection_data(trajectory_analysis: Dict) -> Dict:
        """
        Extract intersection information from trajectory analysis
        
        Args:
            trajectory_analysis: Results from TrajectoryPredictor
            
        Returns:
            Dictionary with intersection data
        """
        intersection = trajectory_analysis.get('intersection', {})
        
        return {
            'intersects': intersection.get('intersects', False),
            'impact_point': intersection.get('impact_point'),
            'predicted_y': intersection.get('predicted_y'),
            'stump_range': intersection.get('stump_range', (0, 0))
        }
    
    @staticmethod
    def extract_quality_metrics(trajectory_analysis: Dict, 
                                tracking_info: Dict) -> Dict:
        """
        Extract quality metrics from both analysis and tracking
        
        Args:
            trajectory_analysis: Results from TrajectoryPredictor
            tracking_info: Results from BallTracker
            
        Returns:
            Dictionary with quality metrics
        """
        return {
            'tracking_quality': tracking_info.get('tracking_quality', 0),
            'fit_quality': trajectory_analysis.get('fit_quality', 0),
            'points_tracked': tracking_info.get('points_tracked', 0),
            'frames_tracked': tracking_info.get('frames_tracked', 0),
            'total_frames': tracking_info.get('total_frames', 0)
        }
    
    @staticmethod
    def extract_all(trajectory_analysis: Dict, 
                   tracking_info: Dict) -> Tuple[Dict, Dict]:
        """
        Extract both intersection data and quality metrics
        
        Args:
            trajectory_analysis: Results from TrajectoryPredictor
            tracking_info: Results from BallTracker
            
        Returns:
            Tuple of (intersection_data, quality_metrics)
        """
        intersection_data = DataExtractor.extract_intersection_data(trajectory_analysis)
        quality_metrics = DataExtractor.extract_quality_metrics(
            trajectory_analysis, tracking_info
        )
        
        return intersection_data, quality_metrics
