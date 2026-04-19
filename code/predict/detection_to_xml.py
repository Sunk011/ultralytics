#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用supervision将检测结果保存为XML格式
支持PASCAL VOC和自定义XML格式
"""

import cv2
import supervision as sv
from ultralytics import YOLO
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
from typing import List, Optional, Dict, Any


class DetectionToXMLConverter:
    """检测结果转XML格式转换器"""
    
    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        初始化转换器
        
        Args:
            class_names: 类别ID到类别名称的映射字典
        """
        self.class_names = class_names or {
            0: "vehicle",
            1: "person", 
            2: "bicycle",
            3: "motorcycle",
            4: "bus",
            5: "truck"
        }
    
    def create_pascal_voc_xml(self, 
                             image_path: str,
                             detections: sv.Detections,
                             output_path: str,
                             image_width: int,
                             image_height: int,
                             image_depth: int = 3) -> None:
        """
        创建PASCAL VOC格式的XML文件
        
        Args:
            image_path: 图像文件路径
            detections: supervision检测结果
            output_path: 输出XML文件路径
            image_width: 图像宽度
            image_height: 图像高度
            image_depth: 图像通道数
        """
        # 创建根元素
        annotation = ET.Element("annotation")
        
        # 添加文件夹信息
        folder = ET.SubElement(annotation, "folder")
        folder.text = os.path.dirname(image_path) if os.path.dirname(image_path) else "."
        
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
        for i in range(len(detections)):
            class_id = detections.class_id[i] if detections.class_id is not None else 0
            confidence = detections.confidence[i] if detections.confidence is not None else 1.0
            x1, y1, x2, y2 = detections.xyxy[i]
            
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
            
            # 置信度（自定义字段）
            confidence_elem = ET.SubElement(obj, "confidence")
            confidence_elem.text = f"{confidence:.6f}"
            
            # 边界框
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(x1))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(y1))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(x2))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(y2))
        
        # 格式化XML并写入文件
        rough_string = ET.tostring(annotation, 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        print(f"PASCAL VOC XML已保存到: {output_path}")
    
    def create_custom_xml(self,
                         image_path: str,
                         detections: sv.Detections,
                         output_path: str,
                         image_width: int,
                         image_height: int) -> None:
        """
        创建自定义格式的XML文件
        
        Args:
            image_path: 图像文件路径
            detections: supervision检测结果
            output_path: 输出XML文件路径
            image_width: 图像宽度
            image_height: 图像高度
        """
        # 创建根元素
        root = ET.Element("detection_results")
        
        # 添加图像信息
        image_info = ET.SubElement(root, "image_info")
        ET.SubElement(image_info, "filename").text = os.path.basename(image_path)
        ET.SubElement(image_info, "path").text = os.path.abspath(image_path)
        ET.SubElement(image_info, "width").text = str(image_width)
        ET.SubElement(image_info, "height").text = str(image_height)
        
        # 添加检测统计
        stats = ET.SubElement(root, "statistics")
        ET.SubElement(stats, "total_detections").text = str(len(detections))
        
        # 统计各类别数量
        class_counts = {}
        if detections.class_id is not None:
            for class_id in detections.class_id:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        classes_elem = ET.SubElement(stats, "classes")
        for class_id, count in class_counts.items():
            class_elem = ET.SubElement(classes_elem, "class")
            class_elem.set("id", str(class_id))
            class_elem.set("name", self.class_names.get(class_id, f"class_{class_id}"))
            class_elem.set("count", str(count))
        
        # 添加检测对象列表
        detections_elem = ET.SubElement(root, "detections")
        
        for i in range(len(detections)):
            class_id = detections.class_id[i] if detections.class_id is not None else 0
            confidence = detections.confidence[i] if detections.confidence is not None else 1.0
            x1, y1, x2, y2 = detections.xyxy[i]
            
            # 创建检测对象
            detection = ET.SubElement(detections_elem, "detection")
            detection.set("id", str(i))
            
            # 类别信息
            class_elem = ET.SubElement(detection, "class")
            class_elem.set("id", str(class_id))
            class_elem.text = self.class_names.get(class_id, f"class_{class_id}")
            
            # 置信度
            conf_elem = ET.SubElement(detection, "confidence")
            conf_elem.text = f"{confidence:.6f}"
            
            # 边界框
            bbox = ET.SubElement(detection, "bounding_box")
            bbox.set("format", "xyxy")
            ET.SubElement(bbox, "x1").text = f"{x1:.2f}"
            ET.SubElement(bbox, "y1").text = f"{y1:.2f}"
            ET.SubElement(bbox, "x2").text = f"{x2:.2f}"
            ET.SubElement(bbox, "y2").text = f"{y2:.2f}"
            
            # 中心点和尺寸
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            center_elem = ET.SubElement(detection, "center")
            ET.SubElement(center_elem, "x").text = f"{center_x:.2f}"
            ET.SubElement(center_elem, "y").text = f"{center_y:.2f}"
            
            size_elem = ET.SubElement(detection, "size")
            ET.SubElement(size_elem, "width").text = f"{width:.2f}"
            ET.SubElement(size_elem, "height").text = f"{height:.2f}"
            ET.SubElement(size_elem, "area").text = f"{width * height:.2f}"
        
        # 格式化XML并写入文件
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        print(f"自定义XML已保存到: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将YOLO检测结果保存为XML格式')
    parser.add_argument('--model', '-m', type=str, 
                       default="/home/sk/project/ultralytics/code/predict/car_vis_v3.pt",
                       help='YOLO模型路径')
    parser.add_argument('--image', '-i', type=str, default='5.jpg',
                       help='输入图像路径')
    parser.add_argument('--output_dir', '-o', type=str, default='.',
                       help='输出目录')
    parser.add_argument('--conf', '-c', type=float, default=0.2,
                       help='置信度阈值')
    parser.add_argument('--imgsz', type=tuple, default=(1080, 1980),
                       help='推理图像尺寸')
    parser.add_argument('--format', '-f', type=str, 
                       choices=['pascal_voc', 'custom', 'both'], 
                       default='both',
                       help='XML格式类型')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"图像文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和图像
    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    
    print(f"加载图像: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"无法读取图像: {args.image}")
        return
    
    image_height, image_width = image.shape[:2]
    print(f"图像尺寸: {image_width}x{image_height}")
    
    # 运行检测
    print(f"运行检测，置信度阈值: {args.conf}")
    results = model(image, imgsz=args.imgsz, conf=args.conf)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    print(f"检测到 {len(detections)} 个目标")
    
    # 创建转换器
    converter = DetectionToXMLConverter()
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # 保存XML
    if args.format in ['pascal_voc', 'both']:
        pascal_output = os.path.join(args.output_dir, f"{base_name}_pascal_voc.xml")
        converter.create_pascal_voc_xml(
            image_path=args.image,
            detections=detections,
            output_path=pascal_output,
            image_width=image_width,
            image_height=image_height
        )
    
    if args.format in ['custom', 'both']:
        custom_output = os.path.join(args.output_dir, f"{base_name}_custom.xml")
        converter.create_custom_xml(
            image_path=args.image,
            detections=detections,
            output_path=custom_output,
            image_width=image_width,
            image_height=image_height
        )
    
    # 可选：保存带标注的图像
    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    
    annotated_output = os.path.join(args.output_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(annotated_output, annotated_image)
    print(f"带标注的图像已保存到: {annotated_output}")
    
    # 打印检测结果统计
    print("\n检测结果统计:")
    if detections.class_id is not None:
        class_counts = {}
        for class_id in detections.class_id:
            class_name = converter.class_names.get(class_id, f"class_{class_id}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} 个")


if __name__ == "__main__":
    main()
