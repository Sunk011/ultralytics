#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成labelimg兼容的PASCAL VOC XML格式文件
只保存检测框信息，不包含置信度
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
from typing import List, Dict, Tuple, Optional


class LabelImgXMLConverter:
    """为labelimg生成标准PASCAL VOC XML格式转换器"""
    
    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        初始化转换器
        
        Args:
            class_names: 类别ID到类别名称的映射字典
        """
        self.class_names = class_names or {
            0: "car",
            1: "bus", 
            2: "other",
            # 3: "motorcycle",
            # 4: "bus",
            # 5: "truck"
        }
    
    def parse_detection_txt(self, txt_path: str) -> List[Tuple[int, List[float]]]:
        """
        解析检测结果txt文件
        
        Args:
            txt_path: txt文件路径
            
        Returns:
            检测结果列表 [(class_id, [x1, y1, x2, y2]), ...]
        """
        detections = []
        
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 6:
                        class_id = int(parts[0])
                        # 跳过置信度，直接读取边界框坐标
                        x1 = float(parts[2])
                        y1 = float(parts[3])
                        x2 = float(parts[4])
                        y2 = float(parts[5])
                        
                        detections.append((class_id, [x1, y1, x2, y2]))
                        
        except Exception as e:
            print(f"解析txt文件时出错: {e}")
            
        return detections
    
    def create_labelimg_xml(self, 
                           txt_path: str,
                           image_path: str,
                           output_path: str,
                           image_width: int = 1920,
                           image_height: int = 1080,
                           image_depth: int = 3) -> None:
        """
        创建labelimg兼容的PASCAL VOC格式XML文件
        
        Args:
            txt_path: 检测结果txt文件路径
            image_path: 对应的图像文件路径
            output_path: 输出XML文件路径
            image_width: 图像宽度
            image_height: 图像高度
            image_depth: 图像通道数
        """
        # 解析检测结果
        detections = self.parse_detection_txt(txt_path)
        
        if not detections:
            print(f"没有检测到任何目标，跳过XML生成")
            return
        
        # 创建根元素
        annotation = ET.Element("annotation")
        
        # 添加文件夹信息
        folder = ET.SubElement(annotation, "folder")
        folder.text = os.path.dirname(image_path) if os.path.dirname(image_path) else "images"
        
        # 添加文件名
        filename = ET.SubElement(annotation, "filename")
        filename.text = os.path.basename(image_path)
        
        # 添加路径
        path = ET.SubElement(annotation, "path")
        path.text = os.path.abspath(image_path)
        
        # 添加源信息
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"
        
        # 添加图像尺寸信息
        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_width)
        height = ET.SubElement(size, "height")
        height.text = str(image_height)
        depth = ET.SubElement(size, "depth")
        depth.text = str(image_depth)
        
        # 添加分割信息
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"
        
        # 添加检测对象
        for class_id, bbox in detections:
            x1, y1, x2, y2 = bbox
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # 跳过无效的边界框
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 创建对象元素
            obj = ET.SubElement(annotation, "object")
            
            # 类别名称
            name = ET.SubElement(obj, "name")
            name.text = self.class_names.get(class_id, f"class_{class_id}")
            
            # 姿态
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            # 截断
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            # 难度
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            # 边界框
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(round(x1)))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(round(y1)))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(round(x2)))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(round(y2)))
        
        # 格式化XML并写入文件
        rough_string = ET.tostring(annotation, 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="    ")
        
        # 移除第一行的XML声明，labelimg有时候对此敏感
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        clean_xml = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(clean_xml)
        
        print(f"LabelImg兼容的XML已保存到: {output_path}")
        print(f"包含 {len(detections)} 个检测目标")
    
    def print_detection_stats(self, txt_path: str) -> None:
        """
        打印检测结果统计信息
        
        Args:
            txt_path: txt文件路径
        """
        detections = self.parse_detection_txt(txt_path)
        
        if not detections:
            print("没有检测到任何目标")
            return
        
        # 统计各类别数量
        class_counts = {}
        
        for class_id, bbox in detections:
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n检测结果统计 (总计: {len(detections)} 个目标):")
        print("-" * 40)
        
        for class_name, count in sorted(class_counts.items()):
            print(f"{class_name:12}: {count:3d} 个目标")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成labelimg兼容的PASCAL VOC XML文件')
    parser.add_argument('--txt_file', '-t', type=str, default='5.txt',
                       help='检测结果txt文件路径')
    parser.add_argument('--image_path', '-i', type=str, default='5.jpg',
                       help='对应的图像文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出XML文件路径（默认根据图像文件名生成）')
    parser.add_argument('--width', '-w', type=int, default=1920,
                       help='图像宽度')
    parser.add_argument('--height', type=int, default=1080,
                       help='图像高度')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.txt_file):
        print(f"txt文件不存在: {args.txt_file}")
        return
    
    # 创建转换器
    converter = LabelImgXMLConverter()
    
    # 显示统计信息
    converter.print_detection_stats(args.txt_file)
    
    # 生成输出文件路径
    if args.output is None:
        # 基于图像文件名生成XML文件名
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        args.output = f"{base_name}.xml"
    
    # 转换为XML
    converter.create_labelimg_xml(
        txt_path=args.txt_file,
        image_path=args.image_path,
        output_path=args.output,
        image_width=args.width,
        image_height=args.height
    )
    
    print(f"\n使用方法:")
    print(f"1. 在labelimg中打开图像文件: {args.image_path}")
    print(f"2. 确保XML文件与图像文件在同一目录: {args.output}")
    print(f"3. labelimg会自动加载对应的XML标注文件")


def quick_convert():
    """快速转换当前目录的5-04.txt文件"""
    converter = LabelImgXMLConverter()
    
    txt_file = "5-04.txt"
    image_file = "5.jpg"
    output_file = "5.xml"
    
    if not os.path.exists(txt_file):
        print(f"文件不存在: {txt_file}")
        return
    
    print("快速转换模式:")
    converter.print_detection_stats(txt_file)
    
    converter.create_labelimg_xml(
        txt_path=txt_file,
        image_path=image_file,
        output_path=output_file,
        image_width=1920,
        image_height=1080
    )
    
    print(f"\n✓ 转换完成！")
    print(f"图像文件: {image_file}")
    print(f"XML文件: {output_file}")
    print(f"现在可以用labelimg打开 {image_file} 查看标注结果")


if __name__ == "__main__":
    import sys
    
    # 如果没有命令行参数，运行快速转换
    if len(sys.argv) == 1:
        quick_convert()
    else:
        main()
