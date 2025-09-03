#!/usr/bin/env python3
"""
YOLO评价工具测试文件
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo_evaluator import YOLOEvaluator

def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建临时目录
    test_dir = Path("test_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    img_dir = test_dir / "images"
    pred_dir = test_dir / "predictions"
    gt_dir = test_dir / "ground_truths"
    
    img_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    
    # 创建测试图像
    image_size = (640, 480)  # width, height
    num_images = 5
    
    for i in range(num_images):
        # 创建测试图像
        img = Image.new('RGB', image_size, color='white')
        img_path = img_dir / f"test_image_{i:03d}.jpg"
        img.save(img_path)
        
        # 创建对应的预测标签 (归一化坐标 + 置信度)
        pred_path = pred_dir / f"test_image_{i:03d}.txt"
        with open(pred_path, 'w') as f:
            # 类别0: 中心在(0.3, 0.3), 大小(0.2, 0.2), 置信度0.8
            f.write("0 0.3 0.3 0.2 0.2 0.8\n")
            # 类别1: 中心在(0.7, 0.7), 大小(0.15, 0.15), 置信度0.9
            f.write("1 0.7 0.7 0.15 0.15 0.9\n")
            if i > 2:  # 后面的图像添加更多预测
                f.write("0 0.5 0.5 0.1 0.1 0.7\n")
        
        # 创建对应的真实标签 (归一化坐标，无置信度)
        gt_path = gt_dir / f"test_image_{i:03d}.txt"
        with open(gt_path, 'w') as f:
            # 类别0: 中心在(0.32, 0.28), 大小(0.18, 0.22) - 与预测接近但有偏差
            f.write("0 0.32 0.28 0.18 0.22\n")
            # 类别1: 中心在(0.68, 0.72), 大小(0.16, 0.14) - 与预测接近但有偏差
            f.write("1 0.68 0.72 0.16 0.14\n")
            if i > 1:  # 后面的图像添加更多真实标签
                f.write("2 0.1 0.1 0.05 0.05\n")
    
    # 创建类别名称文件
    class_names = {
        0: "人",
        1: "车",
        2: "自行车"
    }
    
    with open(test_dir / "class_names.json", 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据已创建在: {test_dir}")
    print(f"  - 图像数量: {num_images}")
    print(f"  - 图像尺寸: {image_size}")
    print(f"  - 类别数量: {len(class_names)}")
    
    return test_dir

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*60)
    print("测试基本功能")
    print("="*60)
    
    # 创建测试数据
    test_dir = create_test_data()
    
    try:
        # 加载类别名称
        with open(test_dir / "class_names.json", 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        
        # 创建评价器
        evaluator = YOLOEvaluator(
            pred_dir=test_dir / "predictions",
            gt_dir=test_dir / "ground_truths", 
            img_dir=test_dir / "images",
            conf_threshold=0.5,
            iou_threshold=0.5,
            class_names=class_names
        )
        
        # 运行评价
        metrics = evaluator.run_evaluation()
        
        if metrics:
            print("\n✅ 基本功能测试通过!")
            print(f"mAP: {metrics['mAP']:.4f}")
            print(f"类别数量: {metrics['num_classes']}")
            
            # 保存结果
            output_dir = test_dir / "results"
            evaluator.save_results(output_dir)
            
            # 生成图表
            evaluator.plot_class_ap_chart(output_dir)
            evaluator.plot_precision_recall_curve(output_dir)
            
            print(f"结果已保存到: {output_dir}")
            
            return True
        else:
            print("❌ 基本功能测试失败!")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试数据
        if test_dir.exists():
            shutil.rmtree(test_dir)

def test_real_data():
    """测试真实数据"""
    print("\n" + "="*60)
    print("测试真实数据")
    print("="*60)
    
    # 检查是否存在真实数据
    pred_dir = Path("asset/pt_labels")
    gt_dir = Path("asset/gt_label") 
    img_dir = Path("asset/images")  # 假设图像在这里
    
    if not all([pred_dir.exists(), gt_dir.exists()]):
        print("❌ 未找到真实数据目录，跳过测试")
        return False
    
    if not img_dir.exists():
        print("⚠️  未找到图像目录，需要指定正确的图像路径")
        return False
    
    try:
        # 创建评价器
        evaluator = YOLOEvaluator(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            img_dir=img_dir,
            conf_threshold=0.25,
            iou_threshold=0.5
        )
        
        # 运行评价
        metrics = evaluator.run_evaluation()
        
        if metrics:
            print("\n✅ 真实数据测试通过!")
            print(f"mAP: {metrics['mAP']:.4f}")
            
            # 保存结果
            output_dir = Path("real_data_results")
            evaluator.save_results(output_dir)
            evaluator.plot_class_ap_chart(output_dir)
            evaluator.plot_precision_recall_curve(output_dir)
            
            # 计算COCO mAP
            coco_map = evaluator.calculate_map_at_multiple_ious()
            print(f"\nCOCO mAP: {coco_map['mAP@[0.5:0.95]']:.4f}")
            
            return True
        else:
            print("❌ 真实数据测试失败!")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边缘情况"""
    print("\n" + "="*60)
    print("测试边缘情况")
    print("="*60)
    
    test_dir = Path("edge_case_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    img_dir = test_dir / "images"
    pred_dir = test_dir / "predictions"
    gt_dir = test_dir / "ground_truths"
    
    img_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    
    try:
        # 测试1: 空预测文件
        img = Image.new('RGB', (640, 480), color='white')
        img.save(img_dir / "empty_pred.jpg")
        
        with open(pred_dir / "empty_pred.txt", 'w') as f:
            pass  # 空文件
        
        with open(gt_dir / "empty_pred.txt", 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        
        # 测试2: 空真实标签文件
        img.save(img_dir / "empty_gt.jpg")
        
        with open(pred_dir / "empty_gt.txt", 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1 0.8\n")
        
        with open(gt_dir / "empty_gt.txt", 'w') as f:
            pass  # 空文件
        
        # 测试3: 低置信度预测
        img.save(img_dir / "low_conf.jpg")
        
        with open(pred_dir / "low_conf.txt", 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1 0.1\n")  # 低置信度
        
        with open(gt_dir / "low_conf.txt", 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        
        # 创建评价器
        evaluator = YOLOEvaluator(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            img_dir=img_dir,
            conf_threshold=0.5,
            iou_threshold=0.5
        )
        
        # 运行评价
        metrics = evaluator.run_evaluation()
        
        print("✅ 边缘情况测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 边缘情况测试失败: {e}")
        return False
    
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)

def main():
    """主测试函数"""
    print("YOLO评价工具测试套件")
    print("="*60)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("基本功能测试", test_basic_functionality()))
    test_results.append(("边缘情况测试", test_edge_cases()))
    # test_results.append(("真实数据测试", test_real_data()))  # 需要真实图像目录
    
    # 输出测试结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了!")
        return True
    else:
        print("⚠️  有测试失败，请检查代码")
        return False

if __name__ == "__main__":
    main()
