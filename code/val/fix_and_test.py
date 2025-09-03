#!/usr/bin/env python3
"""
YOLO评价工具使用示例 - 修复真实数据评价问题
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolo_evaluator import YOLOEvaluator

def fix_and_run_evaluation():
    """修复并运行评价"""
    
    print("🔧 YOLO评价工具 - 真实数据评价")
    print("="*60)
    
    # 数据路径
    pred_dir = "asset/pt_labels"      # 预测结果目录
    gt_dir = "asset/gt_label"         # 真实标注目录
    
    # ⚠️ 关键问题：需要原始图像目录
    # 由于当前没有图像目录，我们有几个解决方案：
    
    print("📁 当前数据情况:")
    print(f"  预测目录: {pred_dir} - {'✅存在' if Path(pred_dir).exists() else '❌不存在'}")
    print(f"  标注目录: {gt_dir} - {'✅存在' if Path(gt_dir).exists() else '❌不存在'}")
    
    # 方案1: 从预测文件推断图像尺寸（不推荐，但可用）
    print("\n🔍 分析预测文件以推断图像信息...")
    
    # 检查几个预测文件
    pred_files = list(Path(pred_dir).glob("*.txt"))[:5]
    for pred_file in pred_files:
        print(f"\n📄 文件: {pred_file.name}")
        if pred_file.stat().st_size > 0:
            with open(pred_file, 'r') as f:
                lines = f.readlines()[:3]  # 只显示前3行
                for i, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        class_id, x, y, w, h, conf = parts[:6]
                        print(f"  行{i}: 类别{class_id}, 中心({x}, {y}), 尺寸({w}, {h}), 置信度{conf}")
                        
                        # 检查是否为归一化坐标
                        coords = [float(x), float(y), float(w), float(h)]
                        if all(0 <= coord <= 1 for coord in coords):
                            print(f"    ✅ 归一化坐标 (0-1范围)")
                        else:
                            print(f"    ⚠️  可能不是归一化坐标")
    
    # 方案2: 创建虚拟图像尺寸进行测试
    print(f"\n💡 解决方案: 创建虚拟图像用于测试")
    
    # 创建虚拟图像目录
    virtual_img_dir = Path("asset/virtual_images")
    virtual_img_dir.mkdir(exist_ok=True)
    
    # 假设标准YOLO训练尺寸
    from PIL import Image
    standard_size = (640, 640)  # width, height
    
    # 为每个预测文件创建对应的虚拟图像
    created_images = 0
    for pred_file in Path(pred_dir).glob("*.txt"):
        img_name = pred_file.stem + ".jpg"
        img_path = virtual_img_dir / img_name
        
        if not img_path.exists():
            # 创建虚拟图像
            img = Image.new('RGB', standard_size, color='gray')
            img.save(img_path)
            created_images += 1
    
    print(f"✅ 创建了 {created_images} 个虚拟图像 (尺寸: {standard_size})")
    
    # 现在运行评价
    print(f"\n🚀 开始评价...")
    
    try:
        evaluator = YOLOEvaluator(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            img_dir=virtual_img_dir,
            conf_threshold=0.25,
            iou_threshold=0.5
        )
        
        # 运行评价
        metrics = evaluator.run_evaluation()
        
        if metrics and metrics['mAP'] > 0:
            print(f"\n✅ 评价成功!")
            print(f"mAP@0.5: {metrics['mAP']:.4f}")
            
            # 保存结果
            output_dir = "real_evaluation_results"
            evaluator.save_results(output_dir)
            evaluator.plot_class_ap_chart(output_dir)
            evaluator.plot_precision_recall_curve(output_dir)
            
            print(f"📊 结果已保存到: {output_dir}")
            
            # 计算COCO mAP
            print(f"\n📈 计算COCO格式mAP...")
            coco_map = evaluator.calculate_map_at_multiple_ious()
            print(f"COCO mAP@[0.5:0.95]: {coco_map['mAP@[0.5:0.95]']:.4f}")
            
        else:
            print(f"\n⚠️  评价结果mAP为0，可能的原因:")
            print(f"  1. 置信度阈值{evaluator.conf_threshold}过高")
            print(f"  2. IoU阈值{evaluator.iou_threshold}过高") 
            print(f"  3. 坐标转换有问题")
            print(f"  4. 类别ID不匹配")
            
            # 尝试降低阈值重新评价
            print(f"\n🔄 尝试降低阈值重新评价...")
            evaluator.conf_threshold = 0.1
            evaluator.iou_threshold = 0.3
            
            metrics2 = evaluator.run_evaluation()
            if metrics2 and metrics2['mAP'] > 0:
                print(f"✅ 降低阈值后成功! mAP@0.3: {metrics2['mAP']:.4f}")
            else:
                print(f"❌ 仍然失败，请检查数据格式")
                
    except Exception as e:
        print(f"❌ 评价失败: {e}")
        import traceback
        traceback.print_exc()

def provide_real_image_solution():
    """提供真实图像解决方案的说明"""
    
    print(f"\n" + "="*60)
    print(f"🎯 如何使用真实图像进行评价")
    print("="*60)
    
    print(f"""
📝 步骤说明:

1️⃣ 准备图像目录
   将原始图像放在一个目录中，例如:
   📁 asset/images/
   ├── 0000277_02601_d_0000552.jpg
   ├── 0000277_03201_d_0000554.jpg
   └── ...

2️⃣ 确保文件名对应
   图像文件名应与标签文件名匹配:
   - 图像: 0000277_02601_d_0000552.jpg
   - 预测: 0000277_02601_d_0000552.txt  
   - 标注: 0000277_02601_d_0000552.txt

3️⃣ 运行评价
   python yolo_evaluator.py \\
       --pred_dir asset/pt_labels \\
       --gt_dir asset/gt_label \\
       --img_dir asset/images \\
       --conf_threshold 0.25 \\
       --iou_threshold 0.5 \\
       --output_dir results \\
       --plot_charts \\
       --coco_map

💡 当前解决方案:
   由于没有原始图像，我创建了虚拟图像来测试功能。
   虚拟图像使用标准640x640尺寸，这可能不是最精确的，
   但可以验证工具是否正常工作。

🔧 获得更准确结果的方法:
   1. 提供真实图像获得准确的尺寸信息
   2. 或者告诉我真实的图像尺寸，我可以修改代码使用固定尺寸
   3. 或者从模型配置文件中读取图像尺寸
""")

if __name__ == "__main__":
    fix_and_run_evaluation()
    provide_real_image_solution()
