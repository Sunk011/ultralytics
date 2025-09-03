#!/usr/bin/env python3
"""
YOLO目标检测评价工具 - 支持归一化标签和原始图像尺寸读取
作者: GitHub Copilot
日期: 2025-08-22
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
import warnings

def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试多种中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
        
        for font_name in chinese_fonts:
            try:
                plt.rcParams['font.family'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                # 测试字体是否可用
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试', fontsize=12)
                plt.close(fig)
                print(f"使用中文字体: {font_name}")
                return True
            except:
                continue
        
        # 如果没有找到中文字体，使用默认字体并警告
        print("警告: 未找到中文字体，使用默认字体")
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False
        
    except Exception as e:
        print(f"字体设置失败: {e}")
        return False

# 初始化字体
setup_chinese_font()

class YOLOEvaluator:
    """YOLO目标检测评价器 - 支持归一化标签"""
    
    def __init__(self, pred_dir, gt_dir, img_dir, conf_threshold=0.25, 
                 iou_threshold=0.5, class_names=None):
        """
        初始化评价器
        
        Args:
            pred_dir: 预测结果目录路径 (归一化标签 + 置信度)
            gt_dir: 真实标注目录路径 (归一化标签)
            img_dir: 原始图像目录路径 (用于获取真实图像尺寸)
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            class_names: 类别名称字典 {class_id: class_name}
        """
        self.pred_dir = Path(pred_dir)
        self.gt_dir = Path(gt_dir)
        self.img_dir = Path(img_dir)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}
        
        # 验证目录存在
        if not self.pred_dir.exists():
            raise ValueError(f"预测目录不存在: {self.pred_dir}")
        if not self.gt_dir.exists():
            raise ValueError(f"标注目录不存在: {self.gt_dir}")
        if not self.img_dir.exists():
            raise ValueError(f"图像目录不存在: {self.img_dir}")
        
        # 存储图像尺寸信息
        self.image_sizes = {}  # {image_name: (width, height)}
        
        # 存储所有预测和真实框
        self.predictions = {}
        self.ground_truths = {}
        
        # 性能指标存储
        self.metrics = {}
        
        print(f"初始化YOLO评价器:")
        print(f"  预测目录: {self.pred_dir}")
        print(f"  标注目录: {self.gt_dir}")
        print(f"  图像目录: {self.img_dir}")
        print(f"  置信度阈值: {self.conf_threshold}")
        print(f"  IoU阈值: {self.iou_threshold}")
    
    def load_image_sizes(self):
        """加载所有图像的尺寸信息"""
        print("正在加载图像尺寸信息...")
        
        # 支持的图像格式
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        image_files = []
        for ext in img_extensions:
            image_files.extend(list(self.img_dir.glob(f"*{ext}")))
            image_files.extend(list(self.img_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            raise ValueError(f"在图像目录中未找到任何图像文件: {self.img_dir}")
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
                    self.image_sizes[img_file.stem] = (width, height)
            except Exception as e:
                print(f"警告: 无法读取图像 {img_file}: {e}")
        
        print(f"成功加载了 {len(self.image_sizes)} 个图像的尺寸信息")
        
        if not self.image_sizes:
            raise ValueError("未能加载任何图像尺寸信息")
    
    def get_image_size(self, image_name):
        """获取图像尺寸"""
        if image_name not in self.image_sizes:
            # 尝试找到匹配的图像
            for img_name in self.image_sizes:
                if img_name.startswith(image_name) or image_name.startswith(img_name):
                    return self.image_sizes[img_name]
            raise ValueError(f"未找到图像 {image_name} 的尺寸信息")
        return self.image_sizes[image_name]
    
    def load_predictions(self):
        """加载预测结果 - 格式: class_id x_center y_center width height confidence"""
        print("正在加载预测结果...")
        
        pred_files = list(self.pred_dir.glob("*.txt"))
        if not pred_files:
            raise ValueError(f"在预测目录中未找到任何txt文件: {self.pred_dir}")
        
        total_predictions = 0
        
        for pred_file in pred_files:
            image_name = pred_file.stem
            predictions = []
            
            if pred_file.stat().st_size > 0:
                try:
                    img_w, img_h = self.get_image_size(image_name)
                except ValueError as e:
                    print(f"警告: 跳过文件 {pred_file.name} - {e}")
                    continue
                
                with open(pred_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            parts = line.strip().split()
                            if len(parts) >= 6:  # 包含置信度
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                confidence = float(parts[5])
                                
                                # 验证归一化坐标范围
                                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                       0 <= width <= 1 and 0 <= height <= 1):
                                    print(f"警告: {pred_file.name} 第{line_num}行坐标可能不是归一化的")
                                
                                # 转换为绝对坐标
                                x1 = (x_center - width/2) * img_w
                                y1 = (y_center - height/2) * img_h
                                x2 = (x_center + width/2) * img_w
                                y2 = (y_center + height/2) * img_h
                                
                                if confidence >= self.conf_threshold:
                                    predictions.append((class_id, confidence, x1, y1, x2, y2))
                                    total_predictions += 1
                        except (ValueError, IndexError) as e:
                            print(f"警告: {pred_file.name} 第{line_num}行格式错误: {line.strip()}")
            
            self.predictions[image_name] = predictions
        
        print(f"加载了 {len(pred_files)} 个预测文件，共 {total_predictions} 个有效预测")
        
        if total_predictions == 0:
            print("警告: 没有找到任何有效的预测结果（可能被置信度阈值过滤）")
    
    def load_ground_truths(self):
        """加载真实标注 - 格式: class_id x_center y_center width height"""
        print("正在加载真实标注...")
        
        gt_files = list(self.gt_dir.glob("*.txt"))
        if not gt_files:
            raise ValueError(f"在标注目录中未找到任何txt文件: {self.gt_dir}")
        
        total_gt = 0
        
        for gt_file in gt_files:
            image_name = gt_file.stem
            ground_truths = []
            
            if gt_file.stat().st_size > 0:
                try:
                    img_w, img_h = self.get_image_size(image_name)
                except ValueError as e:
                    print(f"警告: 跳过文件 {gt_file.name} - {e}")
                    continue
                
                with open(gt_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # 不包含置信度
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                # 验证归一化坐标范围
                                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                                       0 <= width <= 1 and 0 <= height <= 1):
                                    print(f"警告: {gt_file.name} 第{line_num}行坐标可能不是归一化的")
                                
                                # 转换为绝对坐标
                                x1 = (x_center - width/2) * img_w
                                y1 = (y_center - height/2) * img_h
                                x2 = (x_center + width/2) * img_w
                                y2 = (y_center + height/2) * img_h
                                
                                ground_truths.append((class_id, x1, y1, x2, y2))
                                total_gt += 1
                        except (ValueError, IndexError) as e:
                            print(f"警告: {gt_file.name} 第{line_num}行格式错误: {line.strip()}")
            
            self.ground_truths[image_name] = ground_truths
        
        print(f"加载了 {len(gt_files)} 个标注文件，共 {total_gt} 个真实目标")
        
        if total_gt == 0:
            raise ValueError("没有找到任何有效的真实标注")
    
    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 计算并集
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def calculate_ap(self, precisions, recalls):
        """计算AP (使用11点插值法)"""
        # 添加边界点
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # 计算precision的递减包络
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # 11点插值
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p = 0.0
            for i in range(len(recalls)):
                if recalls[i] >= t:
                    p = precisions[i]
                    break
            ap += p / 11
        
        return ap
    
    def evaluate_single_class(self, class_id):
        """评价单个类别"""
        # 收集该类别的所有预测和真实框
        all_predictions = []
        all_ground_truths = []
        num_ground_truths = 0
        
        # 收集预测
        for image_name in self.predictions.keys():
            preds = self.predictions.get(image_name, [])
            for i, (pred_class, conf, x1, y1, x2, y2) in enumerate(preds):
                if pred_class == class_id:
                    all_predictions.append((conf, image_name, i))
        
        # 收集真实框
        for image_name in self.ground_truths.keys():
            gts = self.ground_truths.get(image_name, [])
            for i, (gt_class, x1, y1, x2, y2) in enumerate(gts):
                if gt_class == class_id:
                    all_ground_truths.append((image_name, i))
                    num_ground_truths += 1
        
        if num_ground_truths == 0:
            return 0.0, 0.0, 0.0, [], []
        
        # 按置信度排序
        all_predictions.sort(key=lambda x: x[0], reverse=True)
        
        # 初始化匹配状态
        gt_matched = {}
        for image_name, gt_idx in all_ground_truths:
            if image_name not in gt_matched:
                gt_matched[image_name] = {}
            gt_matched[image_name][gt_idx] = False
        
        # 计算TP和FP
        tp = np.zeros(len(all_predictions))
        fp = np.zeros(len(all_predictions))
        
        for i, (conf, image_name, pred_idx) in enumerate(all_predictions):
            pred_box = self.predictions[image_name][pred_idx][2:6]  # x1, y1, x2, y2
            
            # 找到该图像中相同类别的所有真实框
            gts = self.ground_truths.get(image_name, [])
            gt_boxes = [(j, gt[1:5]) for j, gt in enumerate(gts) if gt[0] == class_id]
            
            max_iou = 0.0
            max_gt_idx = -1
            
            # 找到最大IoU的真实框
            for gt_idx, gt_box in gt_boxes:
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # 判断是否为正确检测
            if max_iou >= self.iou_threshold:
                if image_name in gt_matched and max_gt_idx in gt_matched[image_name]:
                    if not gt_matched[image_name][max_gt_idx]:
                        tp[i] = 1
                        gt_matched[image_name][max_gt_idx] = True
                    else:
                        fp[i] = 1  # 重复检测
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算Precision和Recall
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        recalls = tp_cumsum / num_ground_truths
        
        # 计算AP
        ap = self.calculate_ap(precisions, recalls)
        
        # 返回最终的Precision和Recall
        final_precision = precisions[-1] if len(precisions) > 0 else 0.0
        final_recall = recalls[-1] if len(recalls) > 0 else 0.0
        
        return final_precision, final_recall, ap, precisions, recalls
    
    def evaluate_all_classes(self):
        """评价所有类别"""
        # 获取所有类别
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
        
        print(f"\n开始评价 {len(all_classes)} 个类别...")
        
        for class_id in all_classes:
            precision, recall, ap, precisions, recalls = self.evaluate_single_class(class_id)
            f1_score = 2 * precision * recall / (precision + recall + 1e-16)
            
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            
            class_metrics[class_id] = {
                'class_name': class_name,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'ap': ap,
                'precisions': precisions.tolist(),
                'recalls': recalls.tolist()
            }
            total_ap += ap
            
            print(f"类别 {class_id} ({class_name}): P={precision:.4f}, R={recall:.4f}, AP={ap:.4f}")
        
        map_score = total_ap / len(all_classes) if all_classes else 0.0
        
        self.metrics = {
            'class_metrics': class_metrics,
            'mAP': map_score,
            'num_classes': len(all_classes),
            'iou_threshold': self.iou_threshold,
            'conf_threshold': self.conf_threshold
        }
        
        print(f"\nmAP@{self.iou_threshold}: {map_score:.4f}")
        
        return self.metrics
    
    def plot_class_ap_chart(self, output_dir):
        """绘制各类别AP柱状图"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.metrics:
            print("请先运行评价")
            return
        
        class_ids = []
        class_names = []
        aps = []
        
        for class_id, metrics in self.metrics['class_metrics'].items():
            class_ids.append(class_id)
            class_names.append(f"类别{class_id}")
            aps.append(metrics['ap'])
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(class_ids)), aps, color='skyblue', alpha=0.8)
        
        # 添加数值标签
        for i, (bar, ap) in enumerate(zip(bars, aps)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{ap:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('类别')
        plt.ylabel('平均精度 (AP)')
        plt.title(f'各类别平均精度 (mAP@{self.iou_threshold}={self.metrics["mAP"]:.4f})')
        plt.xticks(range(len(class_ids)), class_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = output_dir / 'class_ap_chart.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"类别AP图表已保存到: {save_path}")
    
    def plot_precision_recall_curve(self, output_dir, class_id=None):
        """绘制PR曲线"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.metrics:
            print("请先运行评价")
            return
        
        plt.figure(figsize=(10, 8))
        
        if class_id is None:
            # 绘制所有类别
            for cid, metrics in self.metrics['class_metrics'].items():
                if len(metrics['recalls']) > 0:
                    class_name = metrics['class_name']
                    ap = metrics['ap']
                    plt.plot(metrics['recalls'], metrics['precisions'], 
                           label=f'{class_name} (AP={ap:.3f})', linewidth=2)
        else:
            # 绘制指定类别
            if class_id in self.metrics['class_metrics']:
                metrics = self.metrics['class_metrics'][class_id]
                class_name = metrics['class_name']
                ap = metrics['ap']
                plt.plot(metrics['recalls'], metrics['precisions'], 
                       label=f'{class_name} (AP={ap:.3f})', linewidth=2)
        
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('精确率-召回率曲线 (PR Curve)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        save_path = output_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PR曲线已保存到: {save_path}")
    
    def calculate_map_at_multiple_ious(self, iou_thresholds=None):
        """计算多个IoU阈值下的mAP (COCO风格)"""
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
        
        map_results = {}
        original_threshold = self.iou_threshold
        
        print("计算多IoU阈值mAP...")
        for iou_thresh in iou_thresholds:
            self.iou_threshold = iou_thresh
            metrics = self.evaluate_all_classes()
            map_results[f'mAP@{iou_thresh:.2f}'] = metrics['mAP']
            print(f"mAP@{iou_thresh:.2f}: {metrics['mAP']:.4f}")
        
        # 计算COCO mAP
        coco_map = np.mean(list(map_results.values()))
        map_results['mAP@[0.5:0.95]'] = coco_map
        
        # 恢复原始阈值
        self.iou_threshold = original_threshold
        
        return map_results
    
    def print_results(self):
        """打印详细结果"""
        if not self.metrics:
            print("请先运行评价")
            return
        
        print("\n" + "="*80)
        print("YOLO目标检测性能评价结果")
        print("="*80)
        
        print(f"IoU阈值: {self.metrics['iou_threshold']}")
        print(f"置信度阈值: {self.metrics['conf_threshold']}")
        print(f"总类别数: {self.metrics['num_classes']}")
        print(f"mAP@{self.metrics['iou_threshold']}: {self.metrics['mAP']:.4f}")
        
        print(f"\n各类别详细指标:")
        print("-" * 80)
        print(f"{'类别ID':<8} {'类别名':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AP':<12}")
        print("-" * 80)
        
        for class_id, metrics in self.metrics['class_metrics'].items():
            class_name = metrics['class_name'][:14]  # 限制长度
            print(f"{class_id:<8} {class_name:<15} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {metrics['ap']:<12.4f}")
        
        print("-" * 80)
        
        # 计算平均值
        avg_precision = np.mean([m['precision'] for m in self.metrics['class_metrics'].values()])
        avg_recall = np.mean([m['recall'] for m in self.metrics['class_metrics'].values()])
        avg_f1 = np.mean([m['f1_score'] for m in self.metrics['class_metrics'].values()])
        
        print(f"{'平均':<8} {'':<15} {avg_precision:<12.4f} {avg_recall:<12.4f} "
              f"{avg_f1:<12.4f} {self.metrics['mAP']:<12.4f}")
        print("="*80)
    
    def save_results(self, output_dir):
        """保存详细结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.metrics:
            print("请先运行评价")
            return
        
        # 保存文本结果
        with open(output_dir / "evaluation_results.txt", 'w', encoding='utf-8') as f:
            f.write("YOLO目标检测性能评价结果\n")
            f.write("="*80 + "\n")
            f.write(f"IoU阈值: {self.metrics['iou_threshold']}\n")
            f.write(f"置信度阈值: {self.metrics['conf_threshold']}\n")
            f.write(f"总类别数: {self.metrics['num_classes']}\n")
            f.write(f"mAP@{self.metrics['iou_threshold']}: {self.metrics['mAP']:.4f}\n\n")
            
            f.write("各类别详细指标:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'类别ID':<8} {'类别名':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AP':<12}\n")
            f.write("-" * 80 + "\n")
            
            for class_id, metrics in self.metrics['class_metrics'].items():
                class_name = metrics['class_name'][:14]
                f.write(f"{class_id:<8} {class_name:<15} {metrics['precision']:<12.4f} "
                       f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {metrics['ap']:<12.4f}\n")
        
        # 保存JSON结果
        json_results = {
            'evaluation_params': {
                'iou_threshold': self.metrics['iou_threshold'],
                'conf_threshold': self.metrics['conf_threshold'],
                'pred_dir': str(self.pred_dir),
                'gt_dir': str(self.gt_dir),
                'img_dir': str(self.img_dir)
            },
            'metrics': self.metrics
        }
        
        with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"评价结果已保存到: {output_dir}")
    
    def run_evaluation(self):
        """运行完整评价流程"""
        try:
            print("开始YOLO模型评价流程...")
            
            # 1. 加载图像尺寸
            self.load_image_sizes()
            
            # 2. 加载预测和标注
            self.load_predictions()
            self.load_ground_truths()
            
            # 3. 评价所有类别
            metrics = self.evaluate_all_classes()
            
            # 4. 显示结果
            self.print_results()
            
            return metrics
            
        except Exception as e:
            print(f"评价过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO目标检测评价工具 - 支持归一化标签')
    parser.add_argument('--pred_dir', type=str, required=True, help='预测结果目录路径')
    parser.add_argument('--gt_dir', type=str, required=True, help='真实标注目录路径')
    parser.add_argument('--img_dir', type=str, required=True, help='原始图像目录路径')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='结果保存目录')
    parser.add_argument('--class_names', type=str, help='类别名称文件路径 (JSON格式)')
    parser.add_argument('--plot_charts', action='store_true', help='生成可视化图表')
    parser.add_argument('--coco_map', action='store_true', help='计算COCO格式的mAP')
    
    args = parser.parse_args()
    
    # 加载类别名称
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
    
    # 创建评价器
    evaluator = YOLOEvaluator(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        img_dir=args.img_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        class_names=class_names
    )
    
    # 运行评价
    metrics = evaluator.run_evaluation()
    
    if metrics:
        # 保存结果
        evaluator.save_results(args.output_dir)
        
        # 生成图表
        if args.plot_charts:
            evaluator.plot_class_ap_chart(args.output_dir)
            evaluator.plot_precision_recall_curve(args.output_dir)
        
        # 计算COCO格式mAP
        if args.coco_map:
            map_results = evaluator.calculate_map_at_multiple_ious()
            
            with open(f"{args.output_dir}/coco_map.json", 'w') as f:
                json.dump(map_results, f, indent=2)
            
            print(f"\nCOCO格式mAP结果:")
            print("-" * 40)
            for key, value in map_results.items():
                print(f"{key}: {value:.4f}")
        
        print(f"\n✅ 评价完成! 基础mAP@{args.iou_threshold}: {metrics['mAP']:.4f}")
    else:
        print("❌ 评价失败")


if __name__ == "__main__":
    main()
