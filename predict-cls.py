import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm # 导入tqdm用于显示进度条

def batch_classify_and_save(model_path: str, input_dir: str, output_dir: str):
    """
    使用YOLO分类模型对一个目录下的所有图片进行批量推理，
    并将带有分类结果的图片保存到指定目录。

    Args:
        model_path (str): 训练好的YOLO分类模型 (.pt) 的路径。
        input_dir (str): 包含待推理图片的输入文件夹路径。
        output_dir (str): 保存带有结果的图片的输出文件夹路径。
    """
    # 1. 输入验证和准备
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在于 '{model_path}'")
        return
    if not os.path.isdir(input_dir):
        print(f"错误: 输入文件夹不存在于 '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存在: {output_dir}")

    # 2. 加载模型
    try:
        model = YOLO(model_path)
        class_names = model.names
        print("YOLO分类模型加载成功！")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 3. 查找所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"在 '{input_dir}' 中没有找到任何图片。")
        return
    
    print(f"找到 {len(image_files)} 张图片，开始处理...")

    # 4. 遍历每张图片进行推理和保存
    # 使用tqdm来创建一个漂亮的进度条
    for filename in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            # 执行推理
            results = model(input_path, verbose=False, device= '4') # verbose=False避免打印过多日志
            result = results[0] # 获取第一张图的结果

            # 解析预测结果
            top1_index = result.probs.top1
            top1_confidence = result.probs.top1conf
            predicted_class = class_names[top1_index]

            # 读取图片以便绘制
            image = cv2.imread(input_path)
            
            # 准备要绘制的文本
            text = f"{predicted_class} ({top1_confidence:.2f})"
            
            # --- 使用OpenCV在图片上绘制文本 ---
            # 定义文本样式
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 2
            text_color = (255, 255, 255) # 白色
            
            # 为了让文本更清晰，先绘制一个半透明的背景矩形
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x, text_y = 10, 40 # 文本左上角坐标
            
            # 绘制背景矩形 (左上角和右下角坐标)
            # 使用黑色背景，可以自行修改颜色 (B, G, R)
            bg_color = (0, 128, 0) if "positive" in predicted_class.lower() else (0, 0, 255) # 正样本用绿色，负样本用红色
            cv2.rectangle(image, (text_x, text_y - text_h - baseline), (text_x + text_w, text_y + baseline), bg_color, -1) # -1 表示填充
            
            # 绘制文本
            cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

            # 保存处理后的图片
            cv2.imwrite(output_path, image)

        except Exception as e:
            print(f"\n处理图片 '{filename}' 时发生错误: {e}")
            
    print("\n所有图片处理完成！")


if __name__ == '__main__':
    # --- 使用说明 ---

    # 1. 设置您的模型路径
    #    将其替换为您自己训练的分类模型的 'best.pt' 文件路径。
    MODEL_PATH = '/home/sk/project/jg_project/cls/test3/weights/best.pt' # <--- 修改这里

    # 2. 设置输入文件夹路径
    #    这个文件夹里存放了所有需要进行分类的图片。
    INPUT_FOLDER = '/home/sk/project/inference_results/cropped_bus' # <--- 修改这里

    # 3. 设置输出文件夹路径
    #    处理后的图片将被保存在这里。
    OUTPUT_FOLDER = './inference_results/classified_images' # <--- 修改这里

    # 4. 运行脚本
    batch_classify_and_save(MODEL_PATH, INPUT_FOLDER, OUTPUT_FOLDER)