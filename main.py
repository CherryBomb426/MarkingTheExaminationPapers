"""答题卡识别判卷系统"""

from Reader import ExamGraderFactory


def main():
    """主函数"""
    print("答题卡识别判卷系统")
    print("=" * 40)
    
    # 创建判卷器
    grader = ExamGraderFactory.create_omr_grader()
    
    # 设置参数
    image_path = "images/test_03.png"
    answer_key = {
        0: 1,
        1: 4,
        2: 0,
        3: 3,
        4: 1
    }
    
    try:
        print(f"处理图像: {image_path}")
        result = grader.process_exam(image_path, answer_key)

        # 这里是试卷判卷后的试卷信息
        result.get_exam_info()
        
        # 输出结果
        print("\n判卷结果:")
        print("-" * 40)
        print(result.get_summary())
        
        if result.wrong_questions:
            wrong_nums = [q+1 for q in result.wrong_questions]
            print(f"错题号: {wrong_nums}")
        
        if result.unanswered_questions:
            unanswered_nums = [q+1 for q in result.unanswered_questions]
            print(f"未答题号: {unanswered_nums}")
        
        print(f"正确率: {result.get_accuracy_rate():.1%}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
