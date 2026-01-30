"""枚举类型定义模块"""

from enum import Enum


class ContourSortMethod(Enum):
    """轮廓排序方法枚举"""
    LEFT_TO_RIGHT = "left-to-right"
    RIGHT_TO_LEFT = "right-to-left"
    TOP_TO_BOTTOM = "top-to-bottom"
    BOTTOM_TO_TOP = "bottom-to-top"


class ThresholdMethod(Enum):
    """二值化方法枚举"""
    BINARY = "binary"
    OTSU = "otsu"
    ADAPTIVE_MEAN = "adaptive_mean"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"


class ImageProcessingMethod(Enum):
    """图像处理方法枚举"""
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_BLUR = "median_blur"
    BILATERAL_FILTER = "bilateral_filter"
    NO_BLUR = "no_blur"