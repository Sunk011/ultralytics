#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML标注文件可视化工具
从XML标注文件中读取标注信息，为每个目标绘制带遮罩的检测框
支持PASCAL VOC格式的XML文件
"""

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict


class XMLVisualizer:
    """XML标注结果可视化类"""
    
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
            "vehicle": (241, 123, 66),
            "person": (128, 0, 128),      
            "other": (255, 255, 0),
            "bicycle": (235, 68, 80),      
            "bus": (255, 0, 255),
            "truck": (0, 255, 255),
            "car": (241, 123, 66),
            "pedestrian": (128, 0, 128),
            "bike": (235, 68, 80),
            "class_0": (241, 123, 66),
            "class_1": (128, 0, 128),
            "class_2": (235, 68, 80),
            "class_3": (255, 255, 0),
            "class_4": (255, 0, 255),
            "class_5": (0, 255, 255),
            "class_6": (128, 0, 128),
            "class_7": (255, 165, 0),
            "class_8": (0, 128, 128),
            "class_9": (128, 128, 0),
        }
        
        self.default_color = (0, 255, 0)  # 默认绿色
        
    def _get_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        根据类别名称获取颜色
        
        Args:
            class_name: 类别名称
            
        Returns:
            BGR颜色值
        """
        return self.class_colors.get(class_name.lower(), self.default_color)
    
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
    
    def draw_annotation_with_mask(self, image: np.ndarray, box: List[int], 
                                 class_name: str = "unknown",
                                 color: Optional[Tuple[int, int, int]] = None,
                                 thickness: Optional[int] = None,
                                 alpha: Optional[float] = None,
                                 show_label: bool = True) -> np.ndarray:
        """
        为目标绘制带遮罩的标注框
        
        Args:
            image: 输入图像
            box: 标注框 [x1, y1, x2, y2]
            class_name: 类别名称
            color: 指定颜色，如果为None则根据class_name自动选择
            thickness: 线条粗细
            alpha: 遮罩透明度
            show_label: 是否显示标签
            
        Returns:
            绘制后的图像
        """
        if color is None:
            color = self._get_color(class_name)
        
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
        # if show_label:
        #     label = class_name
        #     label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
        #     # 绘制标签背景
        #     cv2.rectangle(final_image, (x1, y1 - label_size[1] - 10), 
        #                  (x1 + label_size[0], y1), color, -1)
            
        #     # 绘制标签文本
        #     cv2.putText(final_image, label, (x1, y1 - 5), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return final_image
    
    def parse_xml_file(self, xml_path: str) -> Tuple[Dict, List[Tuple[str, List[int]]]]:
        """
        解析XML标注文件
        
        Args:
            xml_path: XML文件路径
            
        Returns:
            (图像信息字典, 标注列表 [(class_name, [x1, y1, x2, y2]), ...])
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 解析图像信息
            image_info = {}
            filename_elem = root.find('filename')
            if filename_elem is not None:
                image_info['filename'] = filename_elem.text
            
            size_elem = root.find('size')
            if size_elem is not None:
                width_elem = size_elem.find('width')
                height_elem = size_elem.find('height')
                depth_elem = size_elem.find('depth')
                
                if width_elem is not None:
                    image_info['width'] = int(width_elem.text)
                if height_elem is not None:
                    image_info['height'] = int(height_elem.text)
                if depth_elem is not None:
                    image_info['depth'] = int(depth_elem.text)
            
            # 解析标注对象
            annotations = []
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None:
                    continue
                
                class_name = name_elem.text
                
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                
                xmin_elem = bndbox.find('xmin')
                ymin_elem = bndbox.find('ymin')
                xmax_elem = bndbox.find('xmax')
                ymax_elem = bndbox.find('ymax')
                
                if all(elem is not None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                    x1 = int(float(xmin_elem.text))
                    y1 = int(float(ymin_elem.text))
                    x2 = int(float(xmax_elem.text))
                    y2 = int(float(ymax_elem.text))
                    
                    annotations.append((class_name, [x1, y1, x2, y2]))
            
            return image_info, annotations
            
        except Exception as e:
            print(f"解析XML文件时出错: {e}")
            return {}, []
    
    def visualize_xml_annotations(self, xml_file: str, 
                                 background_image: Optional[np.ndarray] = None,
                                 output_path: Optional[str] = None,
                                 show_labels: bool = True) -> np.ndarray:
        """
        可视化XML标注结果
        
        Args:
            xml_file: XML标注文件路径
            background_image: 背景图像，如果为None则创建黑色背景或根据XML中的图像路径加载
            output_path: 输出图像路径
            show_labels: 是否显示标签
            
        Returns:
            可视化结果图像
        """
        # 解析XML标注
        image_info, annotations = self.parse_xml_file(xml_file)
        print(f"读取到 {len(annotations)} 个标注目标")
        
        if image_info:
            print(f"图像信息: {image_info}")
        
        # 创建或使用背景图像
        if background_image is None:
            # 尝试根据XML中的文件名加载图像
            if 'filename' in image_info:
                xml_dir = os.path.dirname(xml_file)
                image_path = os.path.join(xml_dir, image_info['filename'])
                
                if os.path.exists(image_path):
                    background_image = cv2.imread(image_path)
                    if background_image is not None:
                        print(f"自动加载图像: {image_path}")
                        self.image_height, self.image_width = background_image.shape[:2]
            
            # 如果仍然没有背景图像，创建黑色背景
            if background_image is None:
                # 使用XML中的尺寸信息或默认尺寸
                width = image_info.get('width', self.image_width)
                height = image_info.get('height', self.image_height)
                image = np.zeros((height, width, 3), dtype=np.uint8)
                print(f"创建黑色背景图像: {width}x{height}")
            else:
                image = background_image.copy()
        else:
            image = background_image.copy()
            self.image_height, self.image_width = image.shape[:2]
        
        # 绘制所有标注结果
        for class_name, box in annotations:
            image = self.draw_annotation_with_mask(image, box, class_name, show_label=show_labels)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"结果已保存到: {output_path}")
        
        return image
    
    def print_xml_stats(self, xml_file: str) -> None:
        """
        打印XML标注统计信息
        
        Args:
            xml_file: XML文件路径
        """
        image_info, annotations = self.parse_xml_file(xml_file)
        
        if not annotations:
            print("没有找到任何标注")
            return
        
        # 统计各类别数量
        class_counts = {}
        for class_name, box in annotations:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\nXML标注统计 (总计: {len(annotations)} 个目标):")
        print("-" * 50)
        
        for class_name, count in sorted(class_counts.items()):
            color = self._get_color(class_name)
            print(f"{class_name:12}: {count:3d} 个目标, 颜色: {color}")


def main():
    """主函数"""
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='XML标注文件可视化工具')
    parser.add_argument('--xml_file', '-x', type=str, default='5.xml',
                       help='XML标注文件路径 (默认: 5.xml)')
    parser.add_argument('--image_path', '-i', type=str, default=None,
                       help='背景图片路径 (可选，会自动从XML中获取)')
    parser.add_argument('--output_path', '-o', type=str, default='xml_visualization.jpg',
                       help='输出图片路径 (默认: xml_visualization.jpg)')
    parser.add_argument('--width', '-w', type=int, default=1920,
                       help='图像宽度 (默认: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='图像高度 (默认: 1080)')
    parser.add_argument('--no_labels', action='store_true',
                       help='不显示标签文本')
    parser.add_argument('--stats_only', '-s', action='store_true',
                       help='只显示统计信息，不生成可视化图像')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = XMLVisualizer(image_width=args.width, image_height=args.height)
    
    # 检查XML文件是否存在
    if not os.path.exists(args.xml_file):
        print(f"XML文件不存在: {args.xml_file}")
        return
    
    # 显示统计信息
    visualizer.print_xml_stats(args.xml_file)
    
    # 如果只要求统计信息，则退出
    if args.stats_only:
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
        
        print(f"手动加载背景图片: {args.image_path}, 尺寸: {background_image.shape[:2]}")
    
    # 可视化XML标注结果
    result_image = visualizer.visualize_xml_annotations(
        xml_file=args.xml_file,
        background_image=background_image,
        output_path=args.output_path,
        show_labels=not args.no_labels
    )


def quick_example():
    """快速示例"""
    print("=== XML标注可视化快速示例 ===")
    
    # 创建可视化器
    visualizer = XMLVisualizer(image_width=1920, image_height=1080)
    
    # 使用默认设置可视化XML标注
    xml_file = "5.xml"
    
    if os.path.exists(xml_file):
        # 显示统计信息
        visualizer.print_xml_stats(xml_file)
        
        # 生成可视化图像
        result_image = visualizer.visualize_xml_annotations(
            xml_file=xml_file,
            output_path="xml_quick_visualization.jpg"
        )
        print("\n✓ 快速可视化完成，结果保存为 xml_quick_visualization.jpg")
    else:
        print(f"XML文件不存在: {xml_file}")


if __name__ == "__main__":
    import sys
    
    # 如果没有命令行参数，运行快速示例
    if len(sys.argv) == 1:
        quick_example()
    else:
        main()
