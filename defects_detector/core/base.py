from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class DefectDetectionBase(ABC):
    """Base interface for defect detection services"""

    @abstractmethod
    def fit(self, data_loader: Any) -> None:
        """Extract features from training data"""
        pass

    @abstractmethod
    def align(self, data_loader: Any) -> Tuple[float, float]:
        """Compute weight and bias for alignment"""
        pass

    @abstractmethod
    def evaluate(self, data_loader: Any) -> Dict[str, float]:
        """Evaluate model on test data"""
        pass

    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict defects in a single sample"""
        pass
