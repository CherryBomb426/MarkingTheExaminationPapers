"""数据模型定义模块"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ExamResult:
    """考试结果数据类"""
    score: float
    correct_count: int
    wrong_count: int
    unanswered_count: int
    wrong_questions: List[int]
    unanswered_questions: List[int]
    total_questions: int
    question_results: List = None
    graded_image: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """数据验证"""
        if self.score < 0 or self.score > 100:
            raise ValueError("分数必须在0-100之间")
        
        if self.correct_count + self.wrong_count + self.unanswered_count != self.total_questions:
            raise ValueError("题目数量统计不一致")
            
        if len(self.wrong_questions) != self.wrong_count:
            raise ValueError("错题数量与错题列表长度不匹配")
            
        if len(self.unanswered_questions) != self.unanswered_count:
            raise ValueError("未答题数量与未答题列表长度不匹配")
    
    def get_accuracy_rate(self) -> float:
        """获取正确率"""
        if self.total_questions == 0:
            return 0.0
        return self.correct_count / self.total_questions
    
    def get_summary(self) -> str:
        """获取结果摘要字符串"""
        return (f"总分: {self.score:.1f}% | "
                f"答对: {self.correct_count}/{self.total_questions} | "
                f"答错: {self.wrong_count} | "
                f"未答: {self.unanswered_count}")
    
    def get_exam_info(self) -> dict:
        """
        获取考试信息字典
        
        Returns:
            dict: 包含所有考试信息的字典
        """
        return {
            "score": self.score,
            "correct_count": self.correct_count,
            "wrong_count": self.wrong_count,
            "unanswered_count": self.unanswered_count,
            "wrong_questions": self.wrong_questions,
            "unanswered_questions": self.unanswered_questions,
            "total_questions": self.total_questions,
            "accuracy_rate": self.get_accuracy_rate(),
            "graded_image_available": self.graded_image is not None
        }