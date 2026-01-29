"""判卷器工厂类"""

from .omr_grader import OMRExamGrader


class ExamGraderFactory:
    """答题卡判卷器工厂类"""
    
    @staticmethod
    def create_omr_grader(**kwargs) -> OMRExamGrader:
        """创建OMR判卷器"""
        return OMRExamGrader(**kwargs)
