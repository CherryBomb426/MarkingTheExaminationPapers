"""接口定义模块"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class ImageProcessorInterface(ABC):
    """图像处理器接口"""
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray, 
                        blur_kernel: Tuple[int, int] = (5, 5),
                        canny_low: int = 75, 
                        canny_high: int = 200) -> np.ndarray:
        """图像预处理"""
        pass
    
    @abstractmethod
    def find_document_contour(self, edged: np.ndarray, min_area: int = 1000) -> np.ndarray:
        """查找答题卡轮廓"""
        pass
    
    @abstractmethod
    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """四点透视变换"""
        pass
    
    @abstractmethod
    def sort_contours(self, contours: List, method) -> Tuple[List, List]:
        """轮廓排序"""
        pass
    
    @abstractmethod
    def apply_threshold(self, image: np.ndarray, method=None) -> np.ndarray:
        """应用二值化"""
        pass