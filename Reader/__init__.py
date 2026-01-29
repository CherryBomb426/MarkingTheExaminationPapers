"""答题卡判卷系统"""

from .models import ExamResult
from .omr_grader import OMRExamGrader
from .factory import ExamGraderFactory

__version__ = "2.0.0"

__all__ = ['ExamResult', 'OMRExamGrader', 'ExamGraderFactory']
