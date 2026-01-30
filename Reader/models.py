"""数据模型定义模块"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ExamResult:
    """考试结果数据类"""
    score: float  # 考试总分 (0-100)
    correct_count: int  # 正确题数
    wrong_count: int  # 错误题数
    unanswered_count: int  # 未答题数
    wrong_questions: List[int]  # 错题编号列表
    unanswered_questions: List[int]  # 未答题目编号列表
    total_questions: int  # 总题数
    question_results: List = None  # 每题详细结果列表(可选)
    graded_image: Optional[np.ndarray] = None  # 评分后的图像数据(可选)
    
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
                f"未答: {self.unanswered_count} | ")
    
    # def save_graded_image(self, file_path: str) -> bool:
    #     """保存评卷图像到指定路径
    #
    #     Args:
    #         file_path: 图像保存路径
    #
    #     Returns:
    #         bool: 保存成功返回True，否则返回False
    #     """
    #     if self.graded_image is not None:
    #         import cv2
    #         return cv2.imwrite(file_path, self.graded_image)
    #     return False
    #
    # def show_graded_image(self):
    #     """显示评卷图像（如果可用）"""
    #     if self.graded_image is not None:
    #         import cv2
    #         cv2.imshow('Graded Image', self.graded_image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     else:
    #         print("没有评卷图像可供显示")
    
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
            "accuracy_rate": self.get_accuracy_rate()
        }