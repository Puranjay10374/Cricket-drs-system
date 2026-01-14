"""
Decision Making Module

Core logic for making OUT/NOT OUT decisions based on trajectory analysis.
Orchestrates data extraction, validation, outcome analysis, and result building.
"""

from typing import Dict, Optional

from ..config import DecisionConfig
from .components import (
    ConfidenceCalculator,
    DecisionValidator,
    ResponseFormatter,
    DataExtractor,
    OutcomeAnalyzer,
    ResultBuilder,
)


class DecisionMaker:
    """
    Makes OUT/NOT OUT decisions based on trajectory analysis
    
    Orchestrates multiple specialized components to make cricket DRS decisions
    """
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        """
        Initialize decision maker with all components
        
        Args:
            config: Optional decision configuration (uses default if not provided)
        """
        self.config = config or DecisionConfig()
        
        # Initialize all specialized components
        self.confidence_calc = ConfidenceCalculator(self.config)
        self.validator = DecisionValidator()
        self.formatter = ResponseFormatter()
        self.data_extractor = DataExtractor()
        self.outcome_analyzer = OutcomeAnalyzer()
        self.result_builder = ResultBuilder(self.confidence_calc)
    
    def make_decision(self, 
                     trajectory_analysis: Dict,
                     tracking_stats: Dict,
                     debug: bool = True) -> Dict:
        """
        Make final OUT/NOT OUT decision
        
        Args:
            trajectory_analysis: Results from TrajectoryPredictor.analyze_trajectory()
            tracking_stats: Results from BallTracker.track_video()
            
        Returns:
            Dictionary containing:
                - decision: "OUT", "NOT OUT", or "INCONCLUSIVE"
                - confidence: Confidence score (0-1)
                - impact_point: (x, y) coordinates if OUT
                - details: Additional information
        """
        # Step 1: Validate inputs
        valid, error = self.validator.validate_all(trajectory_analysis, tracking_stats)
        if not valid:
            return self.formatter.format_inconclusive(error, tracking_stats)
        
        # Step 2: Extract structured data
        intersection_data, quality_metrics = self.data_extractor.extract_all(
            trajectory_analysis, tracking_stats
        )
        
        # Debug logging
        if debug:
            print("\n=== DEBUG: Decision Making ===")
            print(f"Intersection data: {intersection_data}")
            print(f"Predicted Y: {intersection_data.get('predicted_y')}")
            print(f"Stump range: {intersection_data.get('stump_range')}")
            print(f"Intersects: {intersection_data.get('intersects')}")
            print(f"Quality metrics: {quality_metrics}")
        
        # Step 3: Calculate confidence
        confidence = self.confidence_calc.calculate(
            quality_metrics['tracking_quality'],
            quality_metrics['fit_quality'],
            quality_metrics['points_tracked']
        )
        
        # Step 4: Determine outcome (OUT/NOT OUT)
        decision, reason = self.outcome_analyzer.analyze(intersection_data)
        
        # Step 5: Build complete result
        impact_point = intersection_data.get('impact_point')
        
        return self.result_builder.build_decision_result(
            decision=decision,
            confidence=confidence,
            impact_point=impact_point,
            reason=reason,
            quality_metrics=quality_metrics
        )
    
    def format_response(self, decision_result: Dict) -> Dict:
        """
        Format decision for API response (delegates to ResponseFormatter)
        
        Args:
            decision_result: Output from make_decision()
            
        Returns:
            Clean formatted response for API
        """
        return self.formatter.format_decision(decision_result)


# Backward compatibility alias
DRSDecision = DecisionMaker
