# 导入cv相关库
import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw
# 导入依赖包
import hyperlpr3 as lpr3
import os
from glob import glob


def draw_reticle_box(
    image: np.ndarray, 
    box: list[int], 
    percent: float, 
    color: tuple[int, int, int] = (0, 255, 0), 
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制一个在所有宽高比和percent值下都视觉协调的“瞄准框”。

    采用动态插值逻辑：
    - percent较小时，线长更接近于按“短边”计算的结果，确保角框的紧凑感。
    - percent较大时，线长更接近于按“各自边长”计算的结果，确保能平滑地过渡到完整矩形。
    - percent >= 0.5 时，直接绘制完整矩形。

    Args:
        image (np.ndarray): 要在其上绘制的原始OpenCV图像（BGR格式）。
        box (list[int]): 一个包含四个整数的列表 [x1, y1, x2, y2]。
        percent (float): 控制角线长度的百分比。
        color (tuple[int, int, int], optional): 线的颜色 (B, G, R)。
        thickness (int, optional): 线的粗细。

    Returns:
        np.ndarray: 返回一个【新的】、已经绘制好瞄准框的图像。
    """
    # 1. 创建图像副本和参数处理
    display_image = image.copy()
    DEFAULT_MIN_PERCENT = 0.15
    if percent >= 0.5:
        final_percent = 0.5
    elif percent < 0.1:
        final_percent = DEFAULT_MIN_PERCENT
    else:
        final_percent = percent
        
    x1, y1, x2, y2 = box
    
    # --- 2. 【核心】直接绘制完整矩形并返回的逻辑 ---
    if final_percent == 0.5:
        cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
        return display_image

    # --- 3. 【核心】动态插值计算线长 ---
    width = x2 - x1
    height = y2 - y1
    
    # 目标1：按各自边长计算的长度 (当percent=0.5时是正确的)
    target_len_x1 = width * final_percent
    target_len_y1 = height * final_percent
    
    # 目标2：按短边计算的长度 (当percent很小时是更协调的)
    short_side = min(width, height)
    target_len_short_side = short_side * final_percent
    
    # 插值权重：percent越小，权重越偏向于 target_len_short_side
    # percent从0.1 -> 0.5，权重 w 从 1.0 -> 0.0
    # 我们使用 (0.5 - final_percent) / 0.4 作为权重
    # 当 final_percent=0.1, w = 0.4/0.4 = 1.0
    # 当 final_percent=0.5, w = 0/0.4 = 0.0
    interpolation_weight = (0.5 - final_percent) / 0.4
    
    # 最终的线长是两种计算方式的加权平均
    final_len_x = int( (target_len_short_side * interpolation_weight) + (target_len_x1 * (1 - interpolation_weight)) )
    final_len_y = int( (target_len_short_side * interpolation_weight) + (target_len_y1 * (1 - interpolation_weight)) )
    
    # 4. 绘制8条角线
    # --- 左上角 ---
    cv2.line(display_image, (x1, y1), (x1 + final_len_x, y1), color, thickness)
    cv2.line(display_image, (x1, y1), (x1, y1 + final_len_y), color, thickness)

    # --- 右上角 ---
    cv2.line(display_image, (x2, y1), (x2 - final_len_x, y1), color, thickness)
    cv2.line(display_image, (x2, y1), (x2, y1 + final_len_y), color, thickness)
    
    # --- 左下角 ---
    cv2.line(display_image, (x1, y2), (x1 + final_len_x, y2), color, thickness)
    cv2.line(display_image, (x1, y2), (x1, y2 - final_len_y), color, thickness)

    # --- 右下角 ---
    cv2.line(display_image, (x2, y2), (x2 - final_len_x, y2), color, thickness)
    cv2.line(display_image, (x2, y2), (x2, y2 - final_len_y), color, thickness)
    
    return display_image

def process_single_image(image_path, output_dir, catcher, font_ch):
    """处理单张图片并保存结果"""
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告: 无法读取图片 {image_path}，跳过处理")
        return
    
    # 执行识别算法
    results = catcher(image)
    print(f"处理 {os.path.basename(image_path)}: 找到 {len(results)} 个目标")
    
    # 收集识别结果文本
    detection_texts = []
    for code, confidence, type_idx, box in results:
        text = f"{code} ({confidence:.2f})"
        detection_texts.append(text)
        # 绘制边框
        image = draw_reticle_box(image, box, 0.2)
    
    # 在左上角绘制所有检测文本
    if detection_texts:
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        y_position = 10  # 起始Y坐标
        for text in detection_texts:
            # 绘制文本背景
            draw.rectangle([(10, y_position), (10 + 200, y_position + 20)], fill=(0, 0, 0))
            # 绘制文本
            draw.text((15, y_position), text, (255, 255, 255), font=font_ch)
            y_position += 25  # 每行间隔25像素
        image = np.asarray(img_pil)
    
    # 保存处理后的图片
    try:
        # 获取原文件名并构造输出路径
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(output_path, image)
        print(f"已保存结果: {output_path}")
    except Exception as e:
        print(f"错误: 保存 {filename} 失败 - {e}")

def batch_process_images(input_dir, output_dir, font_path):
    """批量处理文件夹中的所有图片"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载中文字体
    try:
        font_ch = ImageFont.truetype(font_path, 20, 0)
    except Exception as e:
        print(f"错误: 加载字体失败 - {e}")
        return
    
    # 实例化识别对象
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_ULTRA)
    
    # 获取文件夹中所有常见图片格式的文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_dir, ext), recursive=False))
    
    if not image_paths:
        print(f"警告: 在 {input_dir} 中未找到任何图片文件")
        return
    
    # 批量处理图片
    print(f"开始处理 {len(image_paths)} 张图片...")
    for img_path in image_paths:
        process_single_image(img_path, output_dir, catcher, font_ch)
    
    print("批量处理完成！")

if __name__ == "__main__":
    # --- 配置路径 ---
    font_path = "../resource/font/platech.ttf"  # 字体文件路径
    input_directory = "./data"  # 要处理的图片所在文件夹
    output_directory = "./data/out"  # 结果保存目录
    
    # 检查字体文件是否存在
    if not os.path.exists(font_path):
        print(f"错误: 字体文件不存在于路径 '{font_path}'")
    # 检查输入文件夹是否存在
    elif not os.path.isdir(input_directory):
        print(f"错误: 输入文件夹 '{input_directory}' 不存在")
    else:
        # 开始批量处理
        batch_process_images(input_directory, output_directory, font_path)
    