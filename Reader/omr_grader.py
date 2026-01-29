"""OMR答题卡判卷器"""

import numpy as np
import cv2
import imutils
from typing import Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OMRExamGrader:
    """OMR答题卡判卷器"""
    
    def __init__(self, min_bubble_size=20, aspect_ratio_range=(0.9, 1.1), 
                 fill_threshold=50, options_per_question=5):
        """初始化判卷器"""
        self.min_bubble_size = min_bubble_size
        self.aspect_ratio_range = aspect_ratio_range
        self.fill_threshold = fill_threshold
        self.options_per_question = options_per_question
    
    def validate_answer_key(self, answer_key: Dict[int, int]) -> bool:
        """验证答案格式"""
        if not isinstance(answer_key, dict):
            return False
        for q, a in answer_key.items():
            if not isinstance(q, int) or q < 0:
                return False
            if not isinstance(a, int) or a < 0 or a > 4:
                return False
        return True
    
    def process_exam(self, image_path: str, answer_key: Dict[int, int]):
        """处理答题卡识别和判卷"""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        if not self.validate_answer_key(answer_key):
            raise ValueError("答案格式无效")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
            
        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        
        # 查找轮廓
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        # 找到答题卡轮廓
        doc_contour = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx.reshape(4, 2)
                break
        
        if doc_contour is None:
            raise ValueError("未找到答题卡轮廓")
        
        # 透视变换
        warped = self._four_point_transform(gray, doc_contour)
        
        return self._grade_exam(warped, answer_key)
    
    def _four_point_transform(self, image, pts):
        """透视变换"""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    def _order_points(self, pts):
        """排序点"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def _grade_exam(self, warped, answer_key):
        """执行判卷"""
        from Reader.models import ExamResult
        
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        question_cnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if (w >= self.min_bubble_size and h >= self.min_bubble_size and 
                self.aspect_ratio_range[0] <= ar <= self.aspect_ratio_range[1]):
                question_cnts.append(c)
        
        if len(question_cnts) == 0:
            raise ValueError("未找到有效的选项轮廓")
        
        # 排序轮廓
        question_cnts = self._sort_contours(question_cnts, "top-to-bottom")[0]
        
        correct_count = 0
        wrong_questions = []
        unanswered_questions = []
        warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        
        for q in range(len(answer_key)):
            start_idx = q * self.options_per_question
            end_idx = start_idx + self.options_per_question
            
            if end_idx > len(question_cnts):
                continue
                
            current_cnts = self._sort_contours(question_cnts[start_idx:end_idx])[0]
            student_answer = self._detect_answer(thresh, current_cnts)
            correct_answer = answer_key[q]
            
            if student_answer is None:
                unanswered_questions.append(q)
                cv2.drawContours(warped_color, [current_cnts[correct_answer]], -1, (0, 255, 255), 3)
            elif student_answer == correct_answer:
                correct_count += 1
                cv2.drawContours(warped_color, [current_cnts[correct_answer]], -1, (0, 255, 0), 3)
            else:
                wrong_questions.append(q)
                cv2.drawContours(warped_color, [current_cnts[student_answer]], -1, (0, 0, 255), 3)
                cv2.drawContours(warped_color, [current_cnts[correct_answer]], -1, (0, 255, 0), 2)
        
        total = len(answer_key)
        score = (correct_count / total) * 100 if total > 0 else 0
        
        return ExamResult(
            score=score,
            correct_count=correct_count,
            wrong_count=len(wrong_questions),
            unanswered_count=len(unanswered_questions),
            wrong_questions=wrong_questions,
            unanswered_questions=unanswered_questions,
            total_questions=total,
            question_results=[],  # 简化版本暂时为空
            graded_image=warped_color
        )
    
    def _sort_contours(self, cnts, method="left-to-right"):
        """排序轮廓"""
        reverse = method in ["right-to-left", "bottom-to-top"]
        i = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                           key=lambda b: b[1][i], reverse=reverse))
        return cnts, boundingBoxes
    
    def _detect_answer(self, thresh, option_cnts):
        """检测学生答案"""
        max_filled = 0
        selected = None
        
        for i, contour in enumerate(option_cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            filled = cv2.countNonZero(mask)
            
            if filled > max_filled:
                max_filled = filled
                selected = i
        
        return selected if max_filled >= self.fill_threshold else None
