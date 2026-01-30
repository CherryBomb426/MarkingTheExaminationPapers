"""答题卡判卷系统"""

from .models import ExamResult
from .omr_grader import OMRExamGrader
from .factory import ExamGraderFactory
from .image_processor import StandardImageProcessor
from .interfaces import ImageProcessorInterface
from .enums import ContourSortMethod, ThresholdMethod, ImageProcessingMethod

__version__ = "2.0.0"

__all__ = [
    'ExamResult', 
    'OMRExamGrader', 
    'ExamGraderFactory',
    'StandardImageProcessor',
    'ImageProcessorInterface',
    'ContourSortMethod',
    'ThresholdMethod',
    'ImageProcessingMethod'
]
