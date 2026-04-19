#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测结果可视化工具
从标注文件中读取检测结果，为每个目标绘制带遮罩的检测框
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class DetectionVisualizer:
    """检测结果可视化类"""
    
    def __init__(self, image_width: int = 1920, image_height: int = 1080):
        """
        初始化可视化器
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
        """
        self.image_width = image_width
        self.image_height = image_height
        self.default_thickness = 2
        self.mmdet_alpha = 0.3  # 遮罩透明度
        self.reticle_percent = 0.5  # 瞄准框百分比
        
        # 为不同类别设置颜色 (BGR格式)
        self.class_colors = {
            0: (241,123, 66),
            # 1: (64, 225, 136),      
            1: (128, 0, 128),      
            2: (235, 68, 80),      
            3: (255, 255, 0),    # 青色 - 类别3
            4: (255, 0, 255),    # 品红色 - 类别4
            5: (0, 255, 255),    # 黄色 - 类别5
            6: (128, 0, 128),    # 紫色 - 类别6
            7: (255, 165, 0),    # 橙色 - 类别7
            8: (0, 128, 128),    # 深青色 - 类别8
            9: (128, 128, 0),    # 橄榄色 - 类别9
        }
        
        self.default_color = (0, 255, 0)  # 默认绿色
        
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        根据类别ID获取颜色
        
        Args:
            class_id: 类别ID
            
        Returns:
            BGR颜色值
        """
        return self.class_colors.get(class_id, self.default_color)
    
    def draw_reticle_box(self, image: np.ndarray, box: List[int], 
                        percent: float = 0.5, color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        绘制瞄准框样式的检测框
        
        Args:
            image: 输入图像
            box: 检测框 [x1, y1, x2, y2]
            percent: 瞄准框线条长度百分比
            color: 绘制颜色
            
        Returns:
            绘制后的图像
        """
        if color is None:
            color = self.default_color
            
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # 计算瞄准框线条长度
        h_len = int(width * percent)
        v_len = int(height * percent)
        
        thickness = self.default_thickness
        
        # 绘制四个角的瞄准框
        # 左上角
        cv2.line(image, (x1, y1), (x1 + h_len, y1), color, thickness)
        cv2.line(image, (x1, y1), (x1, y1 + v_len), color, thickness)
        
        # 右上角
        cv2.line(image, (x2, y1), (x2 - h_len, y1), color, thickness)
        cv2.line(image, (x2, y1), (x2, y1 + v_len), color, thickness)
        
        # 左下角
        cv2.line(image, (x1, y2), (x1 + h_len, y2), color, thickness)
        cv2.line(image, (x1, y2), (x1, y2 - v_len), color, thickness)
        
        # 右下角
        cv2.line(image, (x2, y2), (x2 - h_len, y2), color, thickness)
        cv2.line(image, (x2, y2), (x2, y2 - v_len), color, thickness)
        
        return image
    
    def draw_detection_with_mask(self, image: np.ndarray, box: List[int], 
                                class_id: int = 0, confidence: float = 1.0,
                                color: Optional[Tuple[int, int, int]] = None,
                                thickness: Optional[int] = None,
                                alpha: Optional[float] = None) -> np.ndarray:
        """
        为目标绘制带遮罩的检测框
        
        Args:
            image: 输入图像
            box: 检测框 [x1, y1, x2, y2]
            class_id: 类别ID
            confidence: 置信度
            color: 指定颜色，如果为None则根据class_id自动选择
            thickness: 线条粗细
            alpha: 遮罩透明度
            
        Returns:
            绘制后的图像
        """
        if color is None:
            color = self._get_color(class_id)
        
        if thickness is None:
            thickness = self.default_thickness
        if alpha is None:
            alpha = self.mmdet_alpha
            
        # 创建一个用于绘制的覆盖层，与原图大小相同
        overlay = image.copy()
        final_image = image.copy()  # 用于最终混合
        x1, y1, x2, y2 = map(int, box)
        
        # --- 核心绘制逻辑 ---
        # 1. 在覆盖层上绘制半透明的填充矩形
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # 2. 绘制类似于瞄准框的检测框
        final_image = self.draw_reticle_box(final_image, [x1, y1, x2, y2], 
                                           percent=self.reticle_percent, color=color)
        
        # 3. 混合遮罩和原图
        final_image = cv2.addWeighted(overlay, alpha, final_image, 1 - alpha, 0)
        
        # # 4. 添加标签文本
        # label = f"Class{class_id}: {confidence:.2f}"
        # label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # # 绘制标签背景
        # cv2.rectangle(final_image, (x1, y1 - label_size[1] - 10), 
        #              (x1 + label_size[0], y1), color, -1)
        
        # # 绘制标签文本
        # cv2.putText(final_image, label, (x1, y1 - 5), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return final_image
    
    def parse_detection_file(self, file_path: str) -> List[Tuple[int, float, List[float]]]:
        """
        解析检测结果文件
        
        Args:
            file_path: 检测结果文件路径
            
        Returns:
            检测结果列表 [(class_id, confidence, [x1, y1, x2, y2]), ...]
        """
        detections = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        confidence = float(parts[1])
                        x1 = float(parts[2])
                        y1 = float(parts[3])
                        x2 = float(parts[4])
                        y2 = float(parts[5])
                        
                        detections.append((class_id, confidence, [x1, y1, x2, y2]))
                        
        except Exception as e:
            print(f"解析文件时出错: {e}")
            
        return detections
    
    def visualize_detections(self, detection_file: str, 
                           background_image: Optional[np.ndarray] = None,
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            detection_file: 检测结果文件路径
            background_image: 背景图像，如果为None则创建黑色背景
            output_path: 输出图像路径
            
        Returns:
            可视化结果图像
        """
        # 解析检测结果
        detections = self.parse_detection_file(detection_file)
        print(f"读取到 {len(detections)} 个检测结果")
        
        # 创建或使用背景图像
        if background_image is None:
            image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        else:
            image = background_image.copy()
            self.image_height, self.image_width = image.shape[:2]
        
        # 绘制所有检测结果
        for class_id, confidence, box in detections:
            image = self.draw_detection_with_mask(image, box, class_id, confidence)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"结果已保存到: {output_path}")
        
        return image


def main():
    """主函数"""
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='目标检测结果可视化工具')
    parser.add_argument('--detection_file', '-d', type=str, default='5-04.txt',
                       help='检测结果文件路径 (默认: 5.txt)')
    parser.add_argument('--image_path', '-i', type=str, default='5.jpg',
                       help='背景图片路径 (可选)')
    parser.add_argument('--output_path', '-o', type=str, default='detection_visualization.jpg',
                       help='输出图片路径 (默认: detection_visualization.jpg)')
    parser.add_argument('--width', '-w', type=int, default=1920,
                       help='图像宽度 (默认: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='图像高度 (默认: 1080)')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = DetectionVisualizer(image_width=args.width, image_height=args.height)
    
    # 检测结果文件路径
    detection_file = args.detection_file
    
    # 检查文件是否存在
    if not os.path.exists(detection_file):
        print(f"文件不存在: {detection_file}")
        return
    
    # 加载背景图像
    background_image = None
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"背景图片不存在: {args.image_path}")
            return
        
        background_image = cv2.imread(args.image_path)
        if background_image is None:
            print(f"无法读取背景图片: {args.image_path}")
            return
        
        print(f"已加载背景图片: {args.image_path}, 尺寸: {background_image.shape[:2]}")
    
    # 可视化检测结果
    result_image = visualizer.visualize_detections(
        detection_file=detection_file,
        background_image=background_image,
        output_path=args.output_path
    )
    
    # # 显示结果
    # print("按任意键关闭窗口...")
    # cv2.imshow("Detection Visualization", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 打印类别统计信息
    detections = visualizer.parse_detection_file(detection_file)
    class_counts = {}
    for class_id, confidence, box in detections:
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    print("\n检测结果统计:")
    for class_id, count in sorted(class_counts.items()):
        color = visualizer._get_color(class_id)
        print(f"类别 {class_id}: {count} 个目标, 颜色: {color}")


def simple_example():
    """简单使用示例"""
    print("=== 简单使用示例 ===")
    
    # 创建可视化器
    visualizer = DetectionVisualizer(image_width=1920, image_height=1080)
    
    # 使用默认设置可视化检测结果
    detection_file = "5.txt"
    
    if os.path.exists(detection_file):
        result_image = visualizer.visualize_detections(
            detection_file=detection_file,
            output_path="simple_visualization.jpg"
        )
        print("简单可视化完成，结果保存为 simple_visualization.jpg")
    else:
        print(f"检测文件不存在: {detection_file}")


if __name__ == "__main__":
    import sys
    
    # # 如果没有命令行参数，运行简单示例
    # if len(sys.argv) == 1:
    #     simple_example()
    # else:
    main()