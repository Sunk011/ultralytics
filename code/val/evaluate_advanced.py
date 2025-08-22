#!/usr/bin/env python3
"""
增强版YOLO评价工具 - 支持不同图像尺寸和更多功能
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib import font_manager



# --- 1. 定义字体文件的路径 ---
font_path = '/home/sk/project/ultralytics/code/val/simhei.ttf'  # 替换为您的字体文件路径

# --- 2. 将字体文件路径注册到 Matplotlib 的字体列表中 ---
# 这会让 Matplotlib “知道”这个新字体的存在
font_manager.fontManager.addfont(font_path)

# --- 3. 设置为全局默认字体 ---
# 我们需要先获取字体的“名称”，然后用它来设置 rcParams
# FontProperties 可以帮助我们从文件中读取字体名称
font_name = font_manager.FontProperties(fname=font_path).get_name()

# 设置全局字体，这样就不需要在每个绘图元素上单独指定了
plt.rcParams['font.family'] = [font_name]

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


class AdvancedDetectionEvaluator:
    """增强版目标检测评价器"""
    
    def __init__(self, pred_dir, gt_dir, img_dir=None, conf_threshold=0.25, 
                 iou_threshold=0.5, img_size=(640, 640)):
        """
        初始化评价器
        
        Args:
            pred_dir: 预测结果目录路径
            gt_dir: 真实标注目录路径
            img_dir: 图像目录路径 (用于获取真实图像尺寸)
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            img_size: 默认图像尺寸 (width, height)
        """
        self.pred_dir = Path(pred_dir)
        self.gt_dir = Path(gt_dir)
        self.img_dir = Path(img_dir) if img_dir else None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.default_img_size = img_size
        
        # 存储图像尺寸信息
        self.image_sizes = {}  # {image_name: (width, height)}
        
        # 存储所有预测和真实框
        self.predictions = {}
        self.ground_truths = {}
        
        # 性能指标存储
        self.class_names = {}
        self.metrics = {}
    
    def load_image_sizes(self):
        """加载图像尺寸信息"""
        if not self.img_dir or not self.img_dir.exists():
            print(f"图像目录不存在，使用默认尺寸 {self.default_img_size}")
            return
        
        print("正在加载图像尺寸信息...")
        
        # 支持的图像格式
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for img_file in self.img_dir.iterdir():
            if img_file.suffix.lower() in img_extensions:
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                        self.image_sizes[img_file.stem] = (width, height)
                except Exception as e:
                    print(f"警告: 无法读取图像 {img_file}: {e}")
        
        print(f"加载了 {len(self.image_sizes)} 个图像的尺寸信息")
    
    def get_image_size(self, image_name):
        """获取图像尺寸"""
        return self.image_sizes.get(image_name, self.default_img_size)
    
    def load_predictions(self):
        """加载预测结果"""
        print("正在加载预测结果...")
        
        for pred_file in self.pred_dir.glob("*.txt"):
            image_name = pred_file.stem
            predictions = []
            
            if pred_file.stat().st_size > 0:
                img_w, img_h = self.get_image_size(image_name)
                
                with open(pred_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            confidence = float(parts[5])
                            
                            # 转换为绝对坐标
                            x1 = (x_center - width/2) * img_w
                            y1 = (y_center - height/2) * img_h
                            x2 = (x_center + width/2) * img_w
                            y2 = (y_center + height/2) * img_h
                            
                            if confidence >= self.conf_threshold:
                                predictions.append((class_id, confidence, x1, y1, x2, y2))
            
            self.predictions[image_name] = predictions
        
        print(f"加载了 {len(self.predictions)} 个预测文件")
    
    def load_ground_truths(self):
        """加载真实标注"""
        print("正在加载真实标注...")
        
        for gt_file in self.gt_dir.glob("*.txt"):
            image_name = gt_file.stem
            ground_truths = []
            
            if gt_file.stat().st_size > 0:
                img_w, img_h = self.get_image_size(image_name)
                
                with open(gt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            # 转换为绝对坐标
                            x1 = (x_center - width/2) * img_w
                            y1 = (y_center - height/2) * img_h
                            x2 = (x_center + width/2) * img_w
                            y2 = (y_center + height/2) * img_h
                            
                            ground_truths.append((class_id, x1, y1, x2, y2))
            
            self.ground_truths[image_name] = ground_truths
        
        print(f"加载了 {len(self.ground_truths)} 个标注文件")
    
    def calculate_iou(self, box1, box2):
        """计算IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def calculate_map_at_multiple_ious(self, iou_thresholds=None):
        """
        计算多个IoU阈值下的mAP
        
        Args:
            iou_thresholds: IoU阈值列表，默认为[0.5:0.95:0.05]
        
        Returns:
            dict: 各IoU阈值下的mAP
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        map_results = {}
        
        for iou_thresh in iou_thresholds:
            # 临时修改IoU阈值
            original_threshold = self.iou_threshold
            self.iou_threshold = iou_thresh
            
            # 计算当前IoU阈值下的指标
            metrics = self.evaluate_all_classes()
            map_results[f'mAP@{iou_thresh:.2f}'] = metrics['mAP']
            
            # 恢复原始阈值
            self.iou_threshold = original_threshold
        
        # 计算mAP@[0.5:0.95]的平均值
        coco_map = np.mean(list(map_results.values()))
        map_results['mAP@[0.5:0.95]'] = coco_map
        
        return map_results
    
    def evaluate_single_class(self, class_id):
        """评价单个类别 - 与原版相同"""
        # [与原版代码相同，这里省略以节省空间]
        # 收集该类别的所有预测和真实框
        all_predictions = []
        all_ground_truths = []
        num_ground_truths = 0
        
        for image_name in self.predictions.keys():
            preds = self.predictions.get(image_name, [])
            for i, (pred_class, conf, x1, y1, x2, y2) in enumerate(preds):
                if pred_class == class_id:
                    all_predictions.append((conf, image_name, i))
            
            gts = self.ground_truths.get(image_name, [])
            for i, (gt_class, x1, y1, x2, y2) in enumerate(gts):
                if gt_class == class_id:
                    all_ground_truths.append((image_name, i))
                    num_ground_truths += 1
        
        if num_ground_truths == 0:
            return 0.0, 0.0, 0.0
        
        all_predictions.sort(key=lambda x: x[0], reverse=True)
        
        gt_matched = {}
        for image_name, gt_idx in all_ground_truths:
            if image_name not in gt_matched:
                gt_matched[image_name] = {}
            gt_matched[image_name][gt_idx] = False
        
        tp = np.zeros(len(all_predictions))
        fp = np.zeros(len(all_predictions))
        
        for i, (conf, image_name, pred_idx) in enumerate(all_predictions):
            pred_box = self.predictions[image_name][pred_idx][2:6]
            
            gts = self.ground_truths.get(image_name, [])
            gt_boxes = [(j, gt[1:5]) for j, gt in enumerate(gts) if gt[0] == class_id]
            
            max_iou = 0.0
            max_gt_idx = -1
            
            for gt_idx, gt_box in gt_boxes:
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= self.iou_threshold:
                if not gt_matched[image_name][max_gt_idx]:
                    tp[i] = 1
                    gt_matched[image_name][max_gt_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        recalls = tp_cumsum / num_ground_truths
        
        ap = self.calculate_ap(precisions, recalls)
        
        final_precision = precisions[-1] if len(precisions) > 0 else 0.0
        final_recall = recalls[-1] if len(recalls) > 0 else 0.0
        
        return final_precision, final_recall, ap
    
    def calculate_ap(self, precisions, recalls):
        """计算AP - 与原版相同"""
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p = 0.0
            for i in range(len(recalls)):
                if recalls[i] >= t:
                    p = precisions[i]
                    break
            ap += p / 11
        
        return ap
    
    def evaluate_all_classes(self):
        """评价所有类别 - 与原版相同"""
        all_classes = set()
        for preds in self.predictions.values():
            for pred in preds:
                all_classes.add(pred[0])
        for gts in self.ground_truths.values():
            for gt in gts:
                all_classes.add(gt[0])
        
        all_classes = sorted(list(all_classes))
        
        class_metrics = {}
        total_ap = 0.0
        
        for class_id in all_classes:
            precision, recall, ap = self.evaluate_single_class(class_id)
            f1_score = 2 * precision * recall / (precision + recall + 1e-16)
            
            class_metrics[class_id] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'ap': ap
            }
            total_ap += ap
        
        map_score = total_ap / len(all_classes) if all_classes else 0.0
        
        self.metrics = {
            'class_metrics': class_metrics,
            'mAP': map_score,
            'num_classes': len(all_classes)
        }
        
        return self.metrics
    
    def save_detailed_results(self, output_dir):
        """保存详细结果，包括JSON格式"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存基本结果
        self.save_results(output_dir / "evaluation_results.txt")
        
        # 保存JSON格式结果
        json_results = {
            'evaluation_params': {
                'iou_threshold': self.iou_threshold,
                'conf_threshold': self.conf_threshold,
                'pred_dir': str(self.pred_dir),
                'gt_dir': str(self.gt_dir)
            },
            'metrics': self.metrics
        }
        
        with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 计算多IoU阈值的mAP
        map_results = self.calculate_map_at_multiple_ious()
        
        with open(output_dir / "map_multiple_ious.json", 'w', encoding='utf-8') as f:
            json.dump(map_results, f, indent=2)
        
        print(f"详细结果已保存到: {output_dir}")
        
        return {
            'basic_metrics': self.metrics,
            'map_multiple_ious': map_results
        }
    
    def plot_precision_recall_curve(self, output_dir, class_id=None):
        """绘制PR曲线"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果没有指定类别，绘制所有类别
        if class_id is None:
            all_classes = list(self.metrics['class_metrics'].keys())
        else:
            all_classes = [class_id]
        
        plt.figure(figsize=(10, 8))
        
        for cls_id in all_classes:
            # 重新计算该类别的详细数据以获取PR曲线
            precision, recall, ap = self.evaluate_single_class(cls_id)
            plt.plot(recall, precision, label=f'Class {cls_id} (AP={ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PR曲线已保存到: {output_dir / 'precision_recall_curve.png'}")
    
    def run_evaluation(self):
        """运行完整评价流程"""
        try:
            # 加载图像尺寸
            self.load_image_sizes()
            
            # 加载数据
            self.load_predictions()
            self.load_ground_truths()
            
            if not self.predictions:
                print("警告: 未找到预测结果文件")
                return None
            
            if not self.ground_truths:
                print("警告: 未找到真实标注文件")
                return None
            
            # 评价
            metrics = self.evaluate_all_classes()
            
            # 显示结果
            self.print_results()
            
            return metrics
            
        except Exception as e:
            print(f"评价过程中出现错误: {e}")
            return None
    
    def print_results(self):
        """打印结果 - 与原版相同"""
        if not self.metrics:
            print("请先运行评价")
            return
        
        print("\n" + "="*80)
        print("YOLO目标检测性能评价结果")
        print("="*80)
        
        print(f"IoU阈值: {self.iou_threshold}")
        print(f"置信度阈值: {self.conf_threshold}")
        print(f"总类别数: {self.metrics['num_classes']}")
        print(f"mAP@{self.iou_threshold}: {self.metrics['mAP']:.4f}")
        
        print("\n各类别详细指标:")
        print("-" * 80)
        print(f"{'类别ID':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AP':<12}")
        print("-" * 80)
        
        for class_id, metrics in self.metrics['class_metrics'].items():
            print(f"{class_id:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} {metrics['ap']:<12.4f}")
        
        print("-" * 80)
        
        avg_precision = np.mean([m['precision'] for m in self.metrics['class_metrics'].values()])
        avg_recall = np.mean([m['recall'] for m in self.metrics['class_metrics'].values()])
        avg_f1 = np.mean([m['f1_score'] for m in self.metrics['class_metrics'].values()])
        
        print(f"{'平均':<8} {avg_precision:<12.4f} {avg_recall:<12.4f} "
              f"{avg_f1:<12.4f} {self.metrics['mAP']:<12.4f}")
        print("="*80)
    
    def save_results(self, output_path):
        """保存结果 - 与原版相同但增加了图像尺寸信息"""
        if not self.metrics:
            print("请先运行评价")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("YOLO目标检测性能评价结果\n")
            f.write("="*80 + "\n")
            f.write(f"IoU阈值: {self.iou_threshold}\n")
            f.write(f"置信度阈值: {self.conf_threshold}\n")
            f.write(f"总类别数: {self.metrics['num_classes']}\n")
            f.write(f"默认图像尺寸: {self.default_img_size}\n")
            f.write(f"实际图像数量: {len(self.image_sizes)}\n")
            f.write(f"mAP@{self.iou_threshold}: {self.metrics['mAP']:.4f}\n\n")
            
            f.write("各类别详细指标:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'类别ID':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AP':<12}\n")
            f.write("-" * 80 + "\n")
            
            for class_id, metrics in self.metrics['class_metrics'].items():
                f.write(f"{class_id:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                       f"{metrics['f1_score']:<12.4f} {metrics['ap']:<12.4f}\n")
            
            f.write("-" * 80 + "\n")
            
            avg_precision = np.mean([m['precision'] for m in self.metrics['class_metrics'].values()])
            avg_recall = np.mean([m['recall'] for m in self.metrics['class_metrics'].values()])
            avg_f1 = np.mean([m['f1_score'] for m in self.metrics['class_metrics'].values()])
            
            f.write(f"{'平均':<8} {avg_precision:<12.4f} {avg_recall:<12.4f} "
                   f"{avg_f1:<12.4f} {self.metrics['mAP']:<12.4f}\n")
        
        print(f"结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版YOLO目标检测结果评价工具')
    parser.add_argument('--pred_dir', type=str, required=True, help='预测结果目录路径')
    parser.add_argument('--gt_dir', type=str, required=True, help='真实标注目录路径')
    parser.add_argument('--img_dir', type=str, help='图像目录路径 (可选，用于获取真实图像尺寸)')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--img_size', type=int, nargs=2, default=[640, 640], 
                        help='默认图像尺寸 [width height]')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                        help='结果保存目录')
    parser.add_argument('--plot_pr', action='store_true', help='绘制PR曲线')
    parser.add_argument('--coco_map', action='store_true', help='计算COCO格式的mAP')
    
    args = parser.parse_args()
    
    # 创建评价器
    evaluator = AdvancedDetectionEvaluator(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        img_dir=args.img_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        img_size=tuple(args.img_size)
    )
    
    # 运行评价
    metrics = evaluator.run_evaluation()
    
    if metrics:
        # 保存详细结果
        detailed_results = evaluator.save_detailed_results(args.output_dir)
        
        # 绘制PR曲线
        if args.plot_pr:
            evaluator.plot_precision_recall_curve(args.output_dir)
        
        # 计算COCO格式mAP
        if args.coco_map:
            print("\nCOCO格式mAP结果:")
            print("-" * 40)
            for key, value in detailed_results['map_multiple_ious'].items():
                print(f"{key}: {value:.4f}")
        
        print(f"\n评价完成!")
        print(f"基础mAP@{args.iou_threshold}: {metrics['mAP']:.4f}")
    else:
        print("评价失败")


if __name__ == "__main__":
    main()
