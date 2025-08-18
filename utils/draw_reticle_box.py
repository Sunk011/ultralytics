# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# 直接按照比例绘制角线

# import cv2
# import numpy as np
# import os

# def draw_reticle_box(
#     image: np.ndarray,
#     box: list[int],
#     percent: float,
#     color: tuple[int, int, int] = (0, 255, 0),
#     thickness: int = 2
# ) -> np.ndarray:
#     """
#     在图像上绘制一个可调节长度的“瞄准框”（角框）。

#     此函数从矩形的四个角点向内绘制8条短线,形成一个瞄准框效果。
#     角线的长度由'percent'参数相对于矩形边长动态控制。

#     Args:
#         image (np.ndarray): 要在其上绘制的原始OpenCV图像（BGR格式）。
#         box (list[int]): 一个包含四个整数的列表 [x1, y1, x2, y2]，代表左上角和右下角的坐标。
#         percent (float): 控制角线长度的百分比。
#                         - 值在[0.1, 0.5]之间时，按实际百分比绘制。
#                         - 值大于0.5时,按0.5处理（绘制完整矩形）。
#                         - 值小于0.1时,使用默认的最小百分比0.15。
#         color (tuple[int, int, int], optional): 线的颜色 (B, G, R)。默认为绿色 (0, 255, 0)。
#         thickness (int, optional): 线的粗细。默认为 2。

#     Returns:
#         np.ndarray: 返回一个已经绘制好瞄准框的图像。原始图像不会被修改。
#     """
#     # 1. 创建图像副本，以避免修改原始图像
#     display_image = image.copy()

#     # 2. 智能处理percent参数
#     DEFAULT_MIN_PERCENT = 0.1
#     if percent > 0.5:
#         final_percent = 0.5
#     elif percent < 0.1:
#         final_percent = DEFAULT_MIN_PERCENT
#     else:
#         final_percent = percent

#     # 3. 解析坐标并计算框的宽高
#     x1, y1, x2, y2 = box
#     width = x2 - x1
#     height = y2 - y1

#     # 4. 计算角线的长度
#     line_len_x = int(width * final_percent)
#     line_len_y = int(height * final_percent)

#     # 5. 绘制8条角线
#     # --- 左上角 ---
#     cv2.line(display_image, (x1, y1), (x1 + line_len_x, y1), color, thickness) # 水平线
#     cv2.line(display_image, (x1, y1), (x1, y1 + line_len_y), color, thickness) # 垂直线

#     # --- 右上角 ---
#     cv2.line(display_image, (x2, y1), (x2 - line_len_x, y1), color, thickness) # 水平线
#     cv2.line(display_image, (x2, y1), (x2, y1 + line_len_y), color, thickness) # 垂直线

#     # --- 左下角 ---
#     cv2.line(display_image, (x1, y2), (x1 + line_len_x, y2), color, thickness) # 水平线
#     cv2.line(display_image, (x1, y2), (x1, y2 - line_len_y), color, thickness) # 垂直线

#     # --- 右下角 ---
#     cv2.line(display_image, (x2, y2), (x2 - line_len_x, y2), color, thickness) # 水平线
#     cv2.line(display_image, (x2, y2), (x2, y2 - line_len_y), color, thickness) # 垂直线

#     return display_image

# # --- 测试代码 ---
# if __name__ == '__main__':
#     # 1. 准备一张测试图片
#     # 如果您有自己的图片，请替换下面的路径
#     test_image_path = 'path/to/your/test_image.jpg' # <--- 修改这里

#     # 如果路径无效，则创建一个黑色的空白图片用于演示
#     if not os.path.exists(test_image_path):
#         print(f"⚠️ 警告: 测试图片 '{test_image_path}' 不存在。将使用一张黑色背景进行演示。")
#         base_image = np.zeros((600, 800, 3), dtype=np.uint8)
#     else:
#         base_image = cv2.imread(test_image_path)

#     # 2. 定义一个模拟的检测框
#     mock_box = [250, 150, 550, 450]

#     # 3. 演示不同percent值下的效果

#     # --- 效果1: 正常值 (0.2) ---
#     image_p20 = draw_reticle_box(base_image, mock_box, percent=0.2, color=(0, 255, 255)) # 黄色
#     cv2.putText(image_p20, "percent = 0.2", (mock_box[0], mock_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     # --- 效果2: 完整矩形 (percent > 0.5) ---
#     image_p_full = draw_reticle_box(image_p20, mock_box, percent=0.7) # 绿色
#     # (为了演示，我们在同一张图上叠加绘制，所以传入的是image_p20)

#     # --- 效果3: 默认最小值 (percent < 0.1) ---
#     small_box = [50, 50, 200, 120]
#     image_p_min = draw_reticle_box(base_image, small_box, percent=0.05, color=(255, 0, 255)) # 粉色
#     cv2.putText(image_p_min, f"percent=0.05 (uses default {0.15})", (small_box[0], small_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

#     # --- 效果4: 临界值 (0.5) ---
#     large_box = [580, 400, 780, 550]
#     image_p_half = draw_reticle_box(base_image, large_box, percent=0.5, color=(255, 255, 0)) # 青色
#     cv2.putText(image_p_half, "percent = 0.5", (large_box[0], large_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#     # 4. 拼接所有效果图并展示
#     # 为了方便对比，我们将四个效果图拼接在一张大图上
#     final_demo_image = np.hstack([image_p20, image_p_min, image_p_half])

#     # 如果您想单独保存每个效果图，可以取消下面的注释
#     os.makedirs("reticle_box_demo", exist_ok=True)
#     cv2.imwrite("reticle_box_demo/demo_p20.jpg", image_p20)
#     cv2.imwrite("reticle_box_demo/demo_p_min.jpg", image_p_min)
#     cv2.imwrite("reticle_box_demo/demo_p_half.jpg", image_p_half)

#     print("✅ 演示完成！")
#     print("已在 'reticle_box_demo' 文件夹中保存了三个独立的演示效果图。")

#     # cv2.imshow("Reticle Box Demonstration", final_demo_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


# # 长度为(长+宽)/2

# import cv2
# import numpy as np
# import os

# def draw_reticle_box_optimized(
#     image: np.ndarray,
#     box: list[int],
#     percent: float,
#     color: tuple[int, int, int] = (0, 255, 0),
#     thickness: int = 2
# ) -> np.ndarray:
#     """
#     【优化版】在图像上绘制一个视觉协调的“瞄准框”（角框）。

#     此函数基于矩形宽高的【平均值】来计算角线长度，确保在不同宽高比下
#     都能获得视觉上均衡的效果。

#     Args:
#         image (np.ndarray): 要在其上绘制的原始OpenCV图像（BGR格式）。
#         box (list[int]): 一个包含四个整数的列表 [x1, y1, x2, y2]。
#         percent (float): 控制角线长度的百分比。
#                          - 值在[0.1, 0.5]之间时，按实际百分比计算。
#                          - 值大于0.5时，按0.5处理（绘制完整矩形）。
#                          - 值小于0.1时，使用默认的最小百分比0.15。
#         color (tuple[int, int, int], optional): 线的颜色 (B, G, R)。默认为绿色。
#         thickness (int, optional): 线的粗细。默认为 2。

#     Returns:
#         np.ndarray: 返回一个【新的】、已经绘制好瞄准框的图像。
#     """
#     # 1. 创建图像副本
#     display_image = image.copy()

#     # 2. 智能处理percent参数
#     DEFAULT_MIN_PERCENT = 0.15
#     if percent > 0.5:
#         final_percent = 0.5
#     elif percent < 0.1:
#         final_percent = DEFAULT_MIN_PERCENT
#     else:
#         final_percent = percent

#     # 3. 解析坐标并计算宽高
#     x1, y1, x2, y2 = box
#     width = x2 - x1
#     height = y2 - y1

#     # 4. 【核心优化】基于平均边长计算统一的线段长度
#     base_length = (width + height) / 2
#     line_length = int(base_length * final_percent)

#     # 为了在 percent=0.5 时能完美闭合，我们需要确保线长不超过宽高的一半
#     line_len_x = min(line_length, width // 2)
#     line_len_y = min(line_length, height // 2)

#     # 5. 绘制8条角线
#     # --- 左上角 ---
#     cv2.line(display_image, (x1, y1), (x1 + line_len_x, y1), color, thickness)
#     cv2.line(display_image, (x1, y1), (x1, y1 + line_len_y), color, thickness)

#     # --- 右上角 ---
#     cv2.line(display_image, (x2, y1), (x2 - line_len_x, y1), color, thickness)
#     cv2.line(display_image, (x2, y1), (x2, y1 + line_len_y), color, thickness)

#     # --- 左下角 ---
#     cv2.line(display_image, (x1, y2), (x1 + line_len_x, y2), color, thickness)
#     cv2.line(display_image, (x1, y2), (x1, y2 - line_len_y), color, thickness)

#     # --- 右下角 ---
#     cv2.line(display_image, (x2, y2), (x2 - line_len_x, y2), color, thickness)
#     cv2.line(display_image, (x2, y2), (x2, y2 - line_len_y), color, thickness)

#     return display_image

# def draw_reticle_box_original(image: np.ndarray, box: list[int], percent: float, color: tuple, thickness: int) -> np.ndarray:
#     """旧版本函数，用于对比。"""
#     display_image = image.copy()
#     final_percent = max(0.15, min(percent, 0.5)) if percent < 0.1 else min(percent, 0.5)
#     x1, y1, x2, y2 = box
#     width, height = x2 - x1, y2 - y1
#     line_len_x = int(width * final_percent)
#     line_len_y = int(height * final_percent)
#     cv2.line(display_image, (x1, y1), (x1 + line_len_x, y1), color, thickness)
#     cv2.line(display_image, (x1, y1), (x1, y1 + line_len_y), color, thickness)
#     cv2.line(display_image, (x2, y1), (x2 - line_len_x, y1), color, thickness)
#     cv2.line(display_image, (x2, y1), (x2, y1 + line_len_y), color, thickness)
#     cv2.line(display_image, (x1, y2), (x1 + line_len_x, y2), color, thickness)
#     cv2.line(display_image, (x1, y2), (x1, y2 - line_len_y), color, thickness)
#     cv2.line(display_image, (x2, y2), (x2 - line_len_x, y2), color, thickness)
#     cv2.line(display_image, (x2, y2), (x2, y2 - line_len_y), color, thickness)
#     return display_image


# # --- 对比测试代码 ---
# if __name__ == '__main__':
#     # 1. 创建一个黑色背景用于演示
#     canvas_height, canvas_width = 400, 800
#     canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

#     # 2. 定义一个宽高比很大的矩形框
#     wide_box = [50, 150, 750, 250] # 宽度700, 高度100

#     # 3. 使用【旧版本】函数进行绘制
#     original_drawn_image = draw_reticle_box_original(
#         canvas, wide_box, percent=0.2, color=(0, 0, 255), thickness=2 # 红色
#     )
#     cv2.putText(original_drawn_image, "Original Method", (wide_box[0], wide_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # 4. 使用【优化版】函数进行绘制
#     optimized_drawn_image = draw_reticle_box_optimized(
#         canvas, wide_box, percent=0.2, color=(0, 255, 0), thickness=2 # 绿色
#     )
#     cv2.putText(optimized_drawn_image, "Optimized Method (Based on Average)", (wide_box[0], wide_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # 5. 将两个结果上下拼接以便于对比
#     # 在两个图像之间添加一个白色分隔线
#     separator = np.full((10, canvas_width, 3), 255, dtype=np.uint8)
#     comparison_image = np.vstack([original_drawn_image, separator, optimized_drawn_image])

#     # 6. 保存并显示结果
#     output_dir = "reticle_box_comparison"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "comparison_wide_box.jpg")
#     cv2.imwrite(output_path, comparison_image)

#     print("✅ 对比演示完成！")
#     print(f"结果已保存至 '{output_path}'。请打开图片查看优化效果。")

#     # cv2.imshow("Comparison: Original (Top, Red) vs Optimized (Bottom, Green)", comparison_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


"""
根据percent的值动态地在这两者之间进行平滑过渡（插值）。
当 percent 很小时 (例如 0.1): 我们更倾向于让所有线段的长度都等于按“短边”计算出的长度。这样可以确保即使在宽矩形上，垂直线也足够长，形成一个紧凑的角。
当 percent 趋近于 0.5 时: 我们更倾向于让线段长度等于其各自边（宽/高）按百分比计算的长度。这样可以确保在percent=0.5时，框能完美闭合。
当 percent 在两者之间时: 我们在这两种计算方式的结果之间进行线性插值，找到一个“折中”的长度。.
"""

import os

import cv2
import numpy as np


def draw_reticle_box(
    image: np.ndarray, box: list[int], percent: float, color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制一个在所有宽高比和percent值下都视觉协调的“瞄准框”。.

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
    final_len_x = int((target_len_short_side * interpolation_weight) + (target_len_x1 * (1 - interpolation_weight)))
    final_len_y = int((target_len_short_side * interpolation_weight) + (target_len_y1 * (1 - interpolation_weight)))

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


# --- 对比测试代码 ---
# if __name__ == '__main__':
#     canvas_height, canvas_width = 400, 800
#     canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
#     wide_box = [50, 150, 750, 250] # 宽度700, 高度100

#     # 1. 使用【旧的平均法】进行绘制 (作为对比)
#     # (为了简洁，这里直接用代码实现旧方法逻辑)
#     width, height = wide_box[2] - wide_box[0], wide_box[3] - wide_box[1]
#     avg_len = int(((width + height) / 2) * 0.2)
#     old_method_image = canvas.copy()
#     x1, y1, x2, y2 = wide_box
#     cv2.line(old_method_image, (x1, y1), (x1 + avg_len, y1), (0, 0, 255), 2)
#     cv2.line(old_method_image, (x1, y1), (x1, y1 + avg_len), (0, 0, 255), 2)
#     cv2.line(old_method_image, (x2, y1), (x2 - avg_len, y1), (0, 0, 255), 2)
#     cv2.line(old_method_image, (x2, y1), (x2, y1 + avg_len), (0, 0, 255), 2)
#     cv2.line(old_method_image, (x1, y2), (x1 + avg_len, y2), (0, 0, 255), 2)
#     cv2.line(old_method_image, (x1, y2), (x1, y2 - avg_len), (0, 0, 255), 2)
#     cv2.line(old_method_image, (x2, y2), (x2 - avg_len, y2), (0, 0, 255), 2)
#     cv2.line(old_method_image, (x2, y2), (x2, y2 - avg_len), (0, 0, 255), 2)
#     cv2.putText(old_method_image, "Previous Method (Based on Average)", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


#     # 2. 使用【最终的插值法】进行绘制
#     final_image = draw_reticle_box(
#         canvas, wide_box, percent=0.2, color=(0, 255, 0), thickness=2 # 绿色
#     )
#     cv2.putText(final_image, "Final Method (Interpolated)", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # 3. 拼接对比
#     separator = np.full((10, canvas_width, 3), 255, dtype=np.uint8)
#     comparison_image = np.vstack([old_method_image, separator, final_image])

#     # 4. 保存
#     output_dir = "reticle_box_final_comparison"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "final_vs_average.jpg")
#     cv2.imwrite(output_path, comparison_image)

#     print("✅ 最终版对比演示完成！")
#     print(f"结果已保存至 '{output_path}'。")

if __name__ == "__main__":
    # 1. 准备一张测试图片
    # 如果您有自己的图片，请替换下面的路径
    test_image_path = "path/to/your/test_image.jpg"  # <--- 修改这里

    # 如果路径无效，则创建一个黑色的空白图片用于演示
    if not os.path.exists(test_image_path):
        print(f"⚠️ 警告: 测试图片 '{test_image_path}' 不存在。将使用一张黑色背景进行演示。")
        base_image = np.zeros((600, 800, 3), dtype=np.uint8)
    else:
        base_image = cv2.imread(test_image_path)

    # 2. 定义一个模拟的检测框
    mock_box = [250, 150, 550, 450]

    # 3. 演示不同percent值下的效果

    # --- 效果1: 正常值 (0.2) ---
    image_p20 = draw_reticle_box(base_image, mock_box, percent=0.2, color=(0, 255, 255))  # 黄色
    cv2.putText(
        image_p20, "percent = 0.2", (mock_box[0], mock_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    # --- 效果2: 完整矩形 (percent > 0.5) ---
    image_p_full = draw_reticle_box(image_p20, mock_box, percent=0.7)  # 绿色
    # (为了演示，我们在同一张图上叠加绘制，所以传入的是image_p20)

    # --- 效果3: 默认最小值 (percent < 0.1) ---
    small_box = [50, 50, 200, 120]
    image_p_min = draw_reticle_box(base_image, small_box, percent=0.05, color=(255, 0, 255))  # 粉色
    cv2.putText(
        image_p_min,
        f"percent=0.05 (uses default {0.15})",
        (small_box[0], small_box[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )

    # --- 效果4: 临界值 (0.5) ---
    large_box = [580, 400, 780, 550]
    image_p_half = draw_reticle_box(base_image, large_box, percent=0.5, color=(255, 255, 0))  # 青色
    cv2.putText(
        image_p_half,
        "percent = 0.5",
        (large_box[0], large_box[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    # 4. 拼接所有效果图并展示
    # 为了方便对比，我们将四个效果图拼接在一张大图上
    final_demo_image = np.hstack([image_p20, image_p_min, image_p_half])

    # 如果您想单独保存每个效果图，可以取消下面的注释
    os.makedirs("reticle_box_demo", exist_ok=True)
    cv2.imwrite("reticle_box_demo/demo_p20.jpg", image_p20)
    cv2.imwrite("reticle_box_demo/demo_p_min.jpg", image_p_min)
    cv2.imwrite("reticle_box_demo/demo_p_half.jpg", image_p_half)

    print("✅ 演示完成！")
    print("已在 'reticle_box_demo' 文件夹中保存了三个独立的演示效果图。")

    # cv2.imshow("Reticle Box Demonstration", final_demo_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
