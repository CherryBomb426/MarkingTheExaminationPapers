"""图像处理工具模块"""

import numpy as np
import cv2
import imutils
from typing import List, Tuple
import logging

from .interfaces import ImageProcessorInterface
from .enums import ContourSortMethod, ThresholdMethod, ImageProcessingMethod

logger = logging.getLogger(__name__)


class StandardImageProcessor(ImageProcessorInterface):
    """标准图像处理器实现"""
    
    def __init__(self, blur_method: ImageProcessingMethod = ImageProcessingMethod.GAUSSIAN_BLUR,
                 threshold_method: ThresholdMethod = ThresholdMethod.OTSU):
        self.blur_method = blur_method
        self.threshold_method = threshold_method
    
    def preprocess_image(self, image: np.ndarray, 
                        blur_kernel: Tuple[int, int] = (5, 5),
                        canny_low: int = 75, 
                        canny_high: int = 200) -> np.ndarray:
        """图像预处理"""
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")
            
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 应用模糊处理
        if self.blur_method == ImageProcessingMethod.GAUSSIAN_BLUR:
            blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
        elif self.blur_method == ImageProcessingMethod.MEDIAN_BLUR:
            blurred = cv2.medianBlur(gray, blur_kernel[0])
        elif self.blur_method == ImageProcessingMethod.BILATERAL_FILTER:
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        else:
            blurred = gray
            
        # Canny边缘检测
        edged = cv2.Canny(blurred, canny_low, canny_high)
        
        logger.info("图像预处理完成")
        return edged
    
    def find_document_contour(self, edged: np.ndarray, min_area: int = 1000) -> np.ndarray:
        """查找答题卡轮廓"""
        if edged is None or edged.size == 0:
            raise ValueError("输入的边缘图像为空")
            
        # 查找轮廓
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cnts) == 0:
            raise ValueError("未找到任何轮廓")

        # 按面积大小排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        # 遍历轮廓，寻找矩形
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                logger.info(f"找到答题卡轮廓，面积: {area:.0f}")
                return approx.reshape(4, 2)
                
        raise ValueError("未找到矩形答题卡轮廓")
    
    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """四点透视变换"""
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")
            
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # 计算变换后的宽度和高度
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # 定义变换后的目标坐标
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # 计算透视变换矩阵并应用变换
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        logger.info(f"透视变换完成，输出尺寸: {maxWidth}x{maxHeight}")
        return warped
    
    def sort_contours(self, contours: List, method: ContourSortMethod) -> Tuple[List, List]:
        """轮廓排序"""
        reverse = method in [ContourSortMethod.RIGHT_TO_LEFT, ContourSortMethod.BOTTOM_TO_TOP]
        i = 1 if method in [ContourSortMethod.TOP_TO_BOTTOM, ContourSortMethod.BOTTOM_TO_TOP] else 0
        
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                               key=lambda b: b[1][i], reverse=reverse))
        
        logger.debug(f"轮廓排序完成，方法: {method.value}, 数量: {len(contours)}")
        return contours, boundingBoxes
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """对4个点进行排序"""
        if pts.shape != (4, 2):
            raise ValueError(f"需要4个点，但得到了{pts.shape[0]}个点")
            
        rect = np.zeros((4, 2), dtype="float32")
        
        # 左上角：x+y最小的点
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        
        # 右下角：x+y最大的点
        rect[2] = pts[np.argmax(s)]
        
        # 右上角：x-y最小的点
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        
        # 左下角：x-y最大的点
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def apply_threshold(self, image: np.ndarray, method: ThresholdMethod = None) -> np.ndarray:
        """应用二值化"""
        if method is None:
            method = self.threshold_method
            
        if method == ThresholdMethod.OTSU:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        elif method == ThresholdMethod.ADAPTIVE_MEAN:
            thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
        elif method == ThresholdMethod.ADAPTIVE_GAUSSIAN:
            thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
        else:  # BINARY
            _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            
        return thresh
