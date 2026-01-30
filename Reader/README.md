# 答题卡识别与自动判卷系统 - Reader模块

## 概述

`Reader` 模块是基于OpenCV的光学标记识别(OMR)答题卡自动判卷系统的核心部分，专门用于处理标准化考试的涂卡式答题卡。该模块能够自动识别答题卡上的填涂选项，并与标准答案进行比对，最终给出评分结果。

## 目录结构

```
Reader/
├── __init__.py          # 模块入口，导出主要类和接口
├── models.py            # 数据模型定义
├── omr_grader.py        # OMR判卷器核心实现
├── factory.py           # 工厂类，用于创建判卷器实例
└── image_processor.py   # 图像处理工具（预留扩展）
```

## 核心组件

### 1. ExamResult (数据模型)

用于封装判卷结果的数据类。

#### 属性:
- `score: float` - 考试分数（0-100）
- `correct_count: int` - 答对题数
- `wrong_count: int` - 答错题数
- `unanswered_count: int` - 未答题数
- `wrong_questions: List[int]` - 答错题号列表
- `unanswered_questions: List[int]` - 未答题号列表
- `total_questions: int` - 总题数
- `question_results: List` - 题目结果列表（可选）
- `graded_image: Optional[np.ndarray]` - 标注后的图像

#### 方法:
- `get_accuracy_rate() -> float` - 获取正确率
- `get_summary() -> str` - 获取结果摘要
- `get_exam_info() -> dict` - 获取考试信息字典

### 2. OMRExamGrader (核心判卷器)

光学标记识别答题卡判卷器的主要实现。

#### 构造参数:
- `min_bubble_size: int = 20` - 最小气泡尺寸阈值
- `aspect_ratio_range: tuple = (0.9, 1.1)` - 气泡宽高比范围
- `fill_threshold: int = 50` - 填涂检测阈值
- `options_per_question: int = 5` - 每题选项数（A-E）

#### 主要方法:
- `process_exam(image_path, answer_key) -> ExamResult` - 处理答题卡
- `validate_answer_key(answer_key) -> bool` - 验证答案格式

### 3. ExamGraderFactory (工厂类)

用于创建判卷器实例的工厂类。

#### 方法:
- `create_omr_grader(**kwargs) -> OMRExamGrader` - 创建OMR判卷器

## 使用指南

### 基本使用

```python
from Reader import ExamGraderFactory

# 1. 创建判卷器
grader = ExamGraderFactory.create_omr_grader()

# 2. 设置标准答案
answer_key = {
    0: 1,  # 第1题答案是B (0=A, 1=B, 2=C, 3=D, 4=E)
    1: 4,  # 第2题答案是E
    2: 0,  # 第3题答案是A
    3: 3,  # 第4题答案是D
    4: 1   # 第5题答案是B
}

# 3. 处理答题卡
result = grader.process_exam("images/test_03.png", answer_key)

# 4. 查看结果
print(result.get_summary())
print(f"正确率: {result.get_accuracy_rate():.1%}")
print(f"详细信息: {result.get_exam_info()}")
```

### 自定义配置

```python
# 创建高精度判卷器（适用于高分辨率图像）
high_precision_grader = ExamGraderFactory.create_omr_grader(
    min_bubble_size=30,
    aspect_ratio_range=(0.85, 1.15),
    fill_threshold=80
)

# 创建宽松判卷器（适用于低质量图像）
tolerant_grader = ExamGraderFactory.create_omr_grader(
    min_bubble_size=15,
    aspect_ratio_range=(0.7, 1.3),
    fill_threshold=30
)
```

### 批量处理

```python
import os
from Reader import ExamGraderFactory

grader = ExamGraderFactory.create_omr_grader()
answer_key = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# 处理目录中的所有图像
image_dir = "../images/"
results = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)
        try:
            result = grader.process_exam(image_path, answer_key)
            results.append((filename, result))
            print(f"{filename}: {result.get_summary()}")
        except Exception as e:
            print(f"{filename}: 处理失败 - {e}")

# 统计总体结果
if results:
    avg_score = sum(result.score for _, result in results) / len(results)
    print(f"\n平均分数: {avg_score:.1f}%")
```

## 技术实现

### 图像处理流程

1. **图像预处理**
   - 灰度化转换
   - 高斯模糊降噪
   - Canny边缘检测

2. **答题卡定位**
   - 轮廓检测
   - 矩形轮廓筛选
   - 四点透视变换

3. **选项识别**
   - 二值化处理（OTSU自适应阈值）
   - 气泡轮廓检测和过滤
   - 轮廓排序（从上到下，从左到右）

4. **答案检测**
   - 填涂程度计算
   - 阈值判断
   - 答案确定

5. **结果生成**
   - 答案对比
   - 统计计算
   - 图像标注

### 参数调优指南

#### min_bubble_size (最小气泡尺寸)
- **推荐值**: 15-30像素
- **调整原则**: 
  - 图像分辨率高 → 增大值
  - 图像分辨率低 → 减小值
  - 有噪声干扰 → 增大值

#### aspect_ratio_range (宽高比范围)
- **推荐值**: (0.8, 1.2)
- **调整原则**:
  - 气泡形状标准 → 缩小范围如(0.9, 1.1)
  - 气泡形状不规则 → 扩大范围如(0.7, 1.3)

#### fill_threshold (填涂阈值)
- **推荐值**: 30-80像素
- **调整原则**:
  - 填涂较浅 → 减小值
  - 填涂较深 → 增大值
  - 有误判 → 调整值

## 错误处理

### 常见错误及解决方案

1. **FileNotFoundError: 图像文件不存在**
   - 检查文件路径是否正确
   - 确认文件存在且可读

2. **ValueError: 未找到答题卡轮廓**
   - 检查图像质量
   - 确保答题卡完整可见
   - 调整图像预处理参数

3. **ValueError: 未找到有效的选项轮廓**
   - 调整min_bubble_size参数
   - 调整aspect_ratio_range参数
   - 检查图像分辨率

4. **ValueError: 答案格式无效**
   - 确保答案字典格式正确
   - 题号从0开始
   - 答案值在0-4范围内

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 保存中间处理结果
import cv2

grader = ExamGraderFactory.create_omr_grader()
result = grader.process_exam("test.png", answer_key)

# 保存标注后的图像
if result.graded_image is not None:
    cv2.imwrite("graded_result.png", result.graded_image)
```

## 性能优化

### 建议的图像规格
- **分辨率**: 300-600 DPI
- **格式**: PNG, JPG
- **尺寸**: 宽度500-1000像素
- **质量**: 清晰，对比度良好

### 性能提升技巧
1. 使用合适的图像分辨率（过高会影响速度）
2. 预处理图像以提高对比度
3. 批量处理时复用判卷器实例
4. 根据实际情况调整参数以减少误判

## 扩展开发

### 添加新的判卷器类型

```python
class CustomGrader:
    def __init__(self, **kwargs):
        # 自定义初始化
        pass
    
    def process_exam(self, image_path, answer_key):
        # 实现自定义判卷逻辑
        pass

# 在工厂类中添加
class ExamGraderFactory:
    @staticmethod
    def create_custom_grader(**kwargs):
        return CustomGrader(**kwargs)
```

### 自定义结果处理

```python
class DetailedExamResult(ExamResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_time = 0
        self.confidence_scores = []
    
    def get_detailed_report(self):
        # 生成详细报告
        pass
```

## API 参考

### ExamResult 类

```python
@dataclass
class ExamResult:
    score: float
    correct_count: int
    wrong_count: int
    unanswered_count: int
    wrong_questions: List[int]
    unanswered_questions: List[int]
    total_questions: int
    question_results: List = None
    graded_image: Optional[np.ndarray] = None

    def get_accuracy_rate(self) -> float:
        """获取正确率 (0-1)"""
        pass

    def get_summary(self) -> str:
        """获取结果摘要字符串"""
        pass

    def get_exam_info(self) -> dict:
        """获取考试信息字典"""
        pass
```

### OMRExamGrader 类

```python
class OMRExamGrader:
    def __init__(self, min_bubble_size=20, 
                 aspect_ratio_range=(0.9, 1.1), 
                 fill_threshold=50, 
                 options_per_question=5):
        pass

    def process_exam(self, image_path: str, answer_key: Dict[int, int]) -> ExamResult:
        """处理答题卡识别和判卷"""
        pass

    def validate_answer_key(self, answer_key: Dict[int, int]) -> bool:
        """验证答案格式"""
        pass
```

### ExamGraderFactory 类

```python
class ExamGraderFactory:
    @staticmethod
    def create_omr_grader(**kwargs) -> OMRExamGrader:
        """创建OMR判卷器"""
        pass
```

## 版本历史

- **v2.0.0**: 重构架构，采用工厂模式，简化接口
- **v1.0.0**: 初始版本，基本OMR功能

## 许可证

MIT License