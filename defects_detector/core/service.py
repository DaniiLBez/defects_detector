from typing import Dict, Any, Tuple

from defects_detector.core.base import DefectDetectionBase
from defects_detector.core.shape_guided_detector import ShapeGuidedDetector


class DefectDetectionService:
    """Factory service for defect detection"""

    @staticmethod
    def create(detector_type: str, config: Dict[str, Any]) -> DefectDetectionBase:
        """Create defect detector based on type"""
        if detector_type == "shapeguided":
            return ShapeGuidedDetector(config)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")

    def __init__(self, detector: DefectDetectionBase):
        """Initialize service with specific detector implementation"""
        self._detector = detector

    def fit(self, data_loader: Any) -> None:
        """Delegate training to underlying detector"""
        return self._detector.fit(data_loader)

    def align(self, data_loader: Any) -> Tuple[float, float]:
        """Delegate alignment to underlying detector"""
        return self._detector.align(data_loader)

    def evaluate(self, data_loader: Any) -> Dict[str, float]:
        """Delegate evaluation to underlying detector"""
        return self._detector.evaluate(data_loader)

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate prediction to underlying detector"""
        return self._detector.predict(data)