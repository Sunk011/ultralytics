# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, "simhei.ttf")

sys.path.append(current_dir)

from draw_reticle_box import draw_reticle_box


def draw_floating_panel_with_arrow(
    image: np.ndarray,
    source_box: list[int],
    attributes: dict[str, str],
    font_path: str = font_path,
    font_size: int = 20,
    text_color: tuple[int, int, int] = (255, 255, 255),
    panel_color: tuple[int, int, int] = (0, 0, 0),
    border_color: tuple[int, int, int] = (0, 255, 0),
    outline_color: tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """
    根据源框在画面中的位置，智能地在其附近绘制一个带指示箭头的悬浮信息面板。.

    定位逻辑:
    1. 如果源框在画面中央，面板固定在其正上方。
    2. 如果源框在角落/边缘，面板会出现在其对角象限，并有一个向上的视觉偏移，以保证布局美观。

    Args:
        image (np.ndarray): 原始OpenCV图像 (BGR格式)。
        source_box (list[int]): 源检测框的坐标 [x1, y1, x2, y2]。
        attributes (dict[str, str]): 要显示的属性字典 {'键': '值', ...}。
        font_path (str, optional): 中文字体文件的路径。默认为 'simhei.ttf'。
        font_size (int, optional): 字体大小。默认为 20。
        text_color (tuple[int, int, int], optional): 文字颜色 (R, G, B)。默认为白色。
        panel_color (tuple[int, int, int], optional): 面板背景色 (B, G, R)。默认为黑色。
        border_color (tuple[int, int, int], optional): 边框和引出线颜色 (B, G, R)。默认为绿色。
        outline_color (tuple[int, int, int], optional): 面板和箭头的外轮廓颜色 (B, G, R)。默认为白色。
        alpha (float, optional): 面板背景的透明度。默认为 0.6。
        thickness (int, optional): 边框和引出线的粗细。默认为 2。

    Returns:
        np.ndarray: 绘制了最终效果的新图像。
    """
    # ... (第0步和第1步：字体检查和尺寸计算，保持不变) ...
    font = None
    if os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"⚠️ 警告: 无法加载字体文件 '{font_path}'。")
    else:
        print(f"⚠️ 警告: 字体文件 '{font_path}' 不存在。")
    max_text_width = 0
    total_text_height = 0
    line_spacing = int(font_size * 0.4)
    lines = [f"{key}: {value}" for key, value in attributes.items()]
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font) if font else (0, 0, len(line) * font_size // 2, font_size)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(line, font=font) if font else (len(line) * font_size // 2, font_size)
        if w > max_text_width:
            max_text_width = w
        total_text_height += h + line_spacing
    padding = int(font_size * 0.5)
    panel_width = max_text_width + 2 * padding
    panel_height = total_text_height - line_spacing + 2 * padding

    # 2. 智能定位逻辑
    sx1, sy1, sx2, sy2 = source_box
    img_h, img_w = image.shape[:2]
    source_center_x, source_center_y = (sx1 + sx2) // 2, (sy1 + sy2) // 2

    # 定义中央区域的边界
    center_x_min, center_x_max = img_w * 0.25, img_w * 0.75
    center_y_min, center_y_max = img_h * 0.25, img_h * 0.75

    offset = int(font_size * 0.75)
    arrow_size = int(font_size * 0.5)

    is_in_center = (center_x_min < source_center_x < center_x_max) and (center_y_min < source_center_y < center_y_max)

    # --- 场景A: 源框在中央 ---
    if is_in_center:
        px = source_center_x - panel_width // 2
        py = sy1 - panel_height - offset
        arrow_points = np.array(
            [
                [px + panel_width // 2, py + panel_height],
                [px + panel_width // 2 - arrow_size, py + panel_height + arrow_size],
                [px + panel_width // 2 + arrow_size, py + panel_height + arrow_size],
            ]
        )
    # --- 场景B: 源框在角落/边缘 ---
    else:
        on_left = source_center_x < img_w / 2
        on_top = source_center_y < img_h / 2

        # 【核心微调】向上偏移量
        y_bias = int(panel_height * 0.5)

        if on_top and on_left:  # 源框在左上 -> 面板在右下 (偏上)
            px = sx2 + offset
            py = sy2 + offset - y_bias
            arrow_points = np.array([[px, py], [px - arrow_size, py], [px, py - arrow_size]])
        elif on_top and not on_left:  # 源框在右上 -> 面板在左下 (偏上)
            px = sx1 - panel_width - offset
            py = sy2 + offset - y_bias
            arrow_points = np.array(
                [[px + panel_width, py], [px + panel_width + arrow_size, py], [px + panel_width, py - arrow_size]]
            )
        elif not on_top and on_left:  # 源框在左下 -> 面板在右上
            px = sx2 + offset
            py = sy1 - panel_height - offset
            arrow_points = np.array(
                [[px, py + panel_height], [px - arrow_size, py + panel_height], [px, py + panel_height + arrow_size]]
            )
        else:  # 源框在右下 -> 面板在左上
            px = sx1 - panel_width - offset
            py = sy1 - panel_height - offset
            arrow_points = np.array(
                [
                    [px + panel_width, py + panel_height],
                    [px + panel_width + arrow_size, py + panel_height],
                    [px + panel_width, py + panel_height + arrow_size],
                ]
            )

    # 确保面板在图像边界内
    px = max(0, min(px, img_w - panel_width))
    py = max(0, min(py, img_h - panel_height))
    px, py = int(px), int(py)

    # ... (第3步和第4步：绘制背景、边框、箭头和文本，与之前版本相同) ...
    overlay = image.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_width, py + panel_height), panel_color, -1)
    outline_thickness = thickness + 2
    cv2.rectangle(overlay, (px, py), (px + panel_width, py + panel_height), outline_color, outline_thickness)
    cv2.rectangle(overlay, (px, py), (px + panel_width, py + panel_height), border_color, thickness)
    cv2.drawContours(overlay, [arrow_points], 0, panel_color, -1)
    cv2.polylines(overlay, [arrow_points], isClosed=True, color=outline_color, thickness=outline_thickness)
    cv2.polylines(overlay, [arrow_points], isClosed=True, color=border_color, thickness=thickness)
    final_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    pil_img = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    current_y = py + padding
    for line in lines:
        draw.text((px + padding, current_y), line, font=font, fill=text_color)
        try:
            bbox = draw.textbbox((0, 0), line, font=font) if font else (0, 0, len(line) * font_size // 2, font_size)
            current_y += (bbox[3] - bbox[1]) + line_spacing
        except AttributeError:
            _, h = draw.textsize(line, font=font) if font else (0, font_size)
            current_y += h + line_spacing
    final_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return final_image


# --- 测试代码 ---
if __name__ == "__main__":
    base_image = np.full((720, 1280, 3), 255, dtype=np.uint8)

    # 定义不同位置的源框
    box_top_left = [50, 50, 250, 200]
    box_bottom_right = [1000, 500, 1200, 650]
    box_center = [540, 260, 740, 460]

    attr = {"ID": "京A·12345", "类型": "公交车", "速度": "45 km/h"}
    car_attributes = {"ID": "沪B·67890", "类型": "私家车", "颜色": "深蓝"}
    car_attributes_2 = {"ID": "沪B·67890", "类型": "私家车", "颜色": "深蓝", "速度": "45 km/h"}

    # 在背景上画出源框
    base_image = draw_reticle_box(base_image, box_top_left, 0.2, (255, 0, 0), 2)
    base_image = draw_reticle_box(base_image, box_bottom_right, 0.1, (0, 255, 0), 2)
    base_image = draw_reticle_box(base_image, box_center, 0.3, (0, 0, 255), 2)

    # --- 演示智能定位 ---
    # 1. 对于左上角的框，面板应出现在右下(偏上)
    annotated_image = draw_floating_panel_with_arrow(base_image, box_top_left, attr, border_color=(255, 0, 0))

    # 2. 对于右下角的框，面板应出现在左上
    annotated_image = draw_floating_panel_with_arrow(
        annotated_image, box_bottom_right, car_attributes, border_color=(0, 255, 0)
    )

    # 3. 对于中间的框，面板应出现在正上方
    annotated_image = draw_floating_panel_with_arrow(
        annotated_image, box_center, car_attributes_2, border_color=(0, 0, 255)
    )

    # --- 保存结果 ---
    output_dir = "floating_panel_demo_smart"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_result.jpg")
    cv2.imwrite(output_path, annotated_image)

    print("✅ 最终智能版演示完成！")
    print(f"结果已保存至 '{output_path}'。")
