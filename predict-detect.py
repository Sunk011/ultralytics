# inference && save

# import cv2
# import numpy as np
# from ultralytics import YOLO

# class YoloDetector:
#     """
#     一个使用YOLOv8模型进行目标检测的工具类。
#     它可以处理视频文件，以mmdetection风格绘制检测框，并保存结果。
#     """

#     def __init__(self, model_path: str = 'yolov8n.pt'):
#         """
#         初始化检测器，加载YOLO模型并为每个类别分配颜色。

#         :param model_path: YOLO模型的路径 (e.g., 'yolov8n.pt'). 
#                            如果模型不存在，ultralytics会自动下载预训练模型。
#         """
#         print(f"正在加载 YOLO 模型: {model_path}")
#         try:
#             self.model = YOLO(model_path)
#             # 获取所有类别名称
#             self.class_names = self.model.names
#             print(f"模型加载成功。检测类别: {list(self.class_names.values())}")

#             # 为每个类别生成一个固定的颜色，用于绘制
#             # 使用随机种子确保每次运行颜色都一样
#             np.random.seed(42) 
#             self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
#         except Exception as e:
#             print(f"错误: 无法加载 YOLO 模型。请检查路径和安装。错误详情: {e}")
#             self.model = None

#     def _draw_mmdet_style_box(self, frame, box, score, class_id):
#         """
#         以mmdetection的风格在图像上绘制单个检测框。
#         这是一个私有辅助方法。

#         :param frame: 要绘制的OpenCV图像帧。
#         :param box: 包含[x1, y1, x2, y2]的边界框坐标。
#         :param score: 检测的置信度。
#         :param class_id: 检测到的类别ID。
#         :return: 绘制了信息的原始frame的副本（用于半透明效果）。
#         """
#         x1, y1, x2, y2 = map(int, box)
#         color = self.colors[class_id].tolist() # 将numpy颜色转为list
#         class_name = self.class_names[class_id]

#         # --- 绘制主边界框 ---
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#         # --- 准备绘制文本和其背景 ---
#         label = f'{class_name} {score:.2f}'
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.7
#         font_thickness = 2
        
#         # 获取文本尺寸
#         (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
#         # 绘制文本背景框 (放在主框的顶部)
#         # 确保背景框不会超出图像顶部
#         label_y1 = max(y1 - text_h - baseline, 0)
#         cv2.rectangle(frame, (x1, label_y1), (x1 + text_w, y1), color, -1) # -1表示填充

#         # --- 绘制文本 ---
#         # 文本颜色应该是对比度高的颜色，这里用白色
#         cv2.putText(frame, label, (x1, y1 - baseline), font, font_scale, (255, 255, 255), font_thickness)

#     def process_video(self, input_path: str, output_path: str, conf_threshold: float = 0.5):
#         """
#         处理输入视频，进行目标检测，并将结果保存到输出视频。

#         :param input_path: 输入视频文件的路径。
#         :param output_path: 保存处理后视频的路径。
#         :param conf_threshold: 置信度阈值，低于此值的检测将被忽略。
#         """
#         if not self.model:
#             print("错误：YOLO模型未初始化，无法处理视频。")
#             return

#         # 打开输入视频
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             print(f"错误: 无法打开输入视频 '{input_path}'")
#             return

#         # 获取视频属性以创建输出视频
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         # 定义视频编码器和创建VideoWriter对象
#         # 使用 'mp4v' 编码器来保存为 .mp4 文件
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#         print(f"开始处理视频... 按 'q' 键可提前终止并保存当前进度。")
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # --- YOLOv8 推理 ---
#             # stream=True 是一个生成器，更省内存
#             results = self.model(frame, stream=True, conf=conf_threshold)

#             # 创建一个副本用于绘制半透明效果
#             overlay = frame.copy()

#             # 遍历检测结果
#             for r in results:
#                 boxes = r.boxes
#                 for box in boxes:
#                     # 获取边界框坐标 (xyxy格式)
#                     bbox = box.xyxy[0]
#                     # 获取置信度
#                     score = float(box.conf[0])
#                     # 获取类别ID
#                     class_id = int(box.cls[0])
                    
#                     # 在 overlay 上绘制
#                     self._draw_mmdet_style_box(overlay, bbox, score, class_id)
            
#             # --- 应用半透明效果 ---
#             # 通过混合原始帧和带有绘制信息的覆盖层来实现
#             alpha = 0.6 # 透明度
#             processed_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

#             # 将处理后的帧写入输出文件
#             out.write(processed_frame)

#             # (可选) 实时显示处理结果
#             # cv2.imshow('YOLOv8 Detection', processed_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("用户手动终止。")
#                 break
        
#         # 释放所有资源
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#         print(f"视频处理完成。结果已保存到 '{output_path}'")


# if __name__ == '__main__':
#     # --- 使用示例 ---

#     # 1. 实例化检测器。
#     # 使用默认的 'yolov8n.pt'。如果本地没有，会自动下载。
#     # 你也可以换成你自己的训练模型 'path/to/your/best.pt'
#     # detector = YoloDetector(model_path=r'/home/sk/project/jg_project/v001/weights/best.pt')
#     # detector = YoloDetector(model_path=r'./ultralytics/yolo11m-obj365-640-Pretrain.pt')
#     detector = YoloDetector(model_path=r'./jg_project/v001-epoch100-aug/weights/best.pt')

#     # 2. 定义输入和输出视频路径
#     # 请将下面一行中的 "path/to/your/input_video.mp4" 替换为您的视频文件实际路径
#     input_video = "./datasets/test-video/720310.mp4"
#     # output_video = "./inference_results/output_detection_video-obj365.mp4"
#     output_video = "./inference_results/output_detection_video-v001-epoch100-aug.mp4"
    
#     # 3. 调用处理方法
#     if detector.model:
#         detector.process_video(input_video, output_video, conf_threshold=0.4)

























# # inference && crop target images
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os

# class YoloDetector:
#     """
#     一个使用YOLOv8模型进行目标检测的工具类。
#     它可以处理视频文件，以mmdetection风格绘制检测框，并能将指定类别的物体裁剪并保存。
#     【新特性】保存的裁剪图片会以持久化的数字序列（1.jpg, 2.jpg, ...）命名。
#     """

#     def __init__(self, model_path: str = 'yolov8n.pt'):
#         """
#         初始化检测器，加载YOLO模型并为每个类别分配颜色。
#         """
#         print(f"正在加载 YOLO 模型: {model_path}")
#         try:
#             self.model = YOLO(model_path)
#             self.class_names = self.model.names
#             print(f"模型加载成功。检测类别: {list(self.class_names.values())}")
#             self.class_name_to_id = {v: k for k, v in self.class_names.items()}

#             np.random.seed(42)
#             self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
            
#             # 【重大修改】为裁剪文件的命名创建一个实例计数器
#             self.crop_file_counter = 0

#         except Exception as e:
#             print(f"错误: 无法加载 YOLO 模型。请检查路径和安装。错误详情: {e}")
#             self.model = None

#     # 【新函数】获取目录中现有图片的最大编号
#     def get_highest_file_number(self, directory: str) -> int:
#         """
#         检查指定目录，如果不存在则创建它。
#         然后扫描目录中所有 '数字.jpg' 格式的文件，并返回其中最大的数字。
#         如果目录为空或没有符合格式的文件，则返回 0。

#         :param directory: 要扫描的文件夹目录。
#         :return: 文件夹中现有图片的最大编号。
#         """
#         # 确保目录存在，如果不存在则创建
#         os.makedirs(directory, exist_ok=True)
        
#         max_num = 0
#         try:
#             files = os.listdir(directory)
#             for filename in files:
#                 # 检查文件是否为jpg格式
#                 if filename.lower().endswith('.jpg'):
#                     # 尝试从文件名中提取数字部分
#                     try:
#                         # 去掉 .jpg 后缀
#                         num_str = os.path.splitext(filename)[0]
#                         num = int(num_str)
#                         if num > max_num:
#                             max_num = num
#                     except ValueError:
#                         # 如果文件名不是纯数字，则忽略该文件
#                         continue
#         except Exception as e:
#             print(f"扫描目录时发生错误: {e}")

#         print(f"扫描目录 '{directory}' 完成。找到的最大文件编号为: {max_num}。")
#         return max_num

#     def _draw_mmdet_style_box(self, frame, box, score, class_id):
#         """
#         以mmdetection的风格在图像上绘制单个检测框。
#         """
#         x1, y1, x2, y2 = map(int, box)
#         color = self.colors[class_id].tolist()
#         class_name = self.class_names[class_id]

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#         label = f'{class_name} {score:.2f}'
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.7
#         font_thickness = 2
#         (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
#         label_y1 = max(y1 - text_h - baseline, 0)
#         cv2.rectangle(frame, (x1, label_y1), (x1 + text_w, y1), color, -1)
#         cv2.putText(frame, label, (x1, y1 - baseline), font, font_scale, (255, 255, 255), font_thickness)


#     def crop_and_save_objects(self, single_frame_result, original_frame, target_class_id: int, save_dir: str):
#         """
#         从单帧检测结果中裁剪出指定类别的物体并以递增数字命名保存。

#         :param single_frame_result: 来自YOLO模型的一帧的检测结果对象。
#         :param original_frame: 原始的OpenCV图像帧，用于裁剪。
#         :param target_class_id: 要裁剪的目标物体的类别ID。
#         :param save_dir: 保存裁剪后图像的目录。
#         """
#         boxes = single_frame_result.boxes
#         original_frame = original_frame.copy()
#         for box in boxes:
#             class_id = int(box.cls[0])
            
#             if class_id == target_class_id:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(original_frame.shape[1], x2), min(original_frame.shape[0], y2)
                
#                 if x1 < x2 and y1 < y2:
#                     cropped_object = original_frame[y1:y2, x1:x2]
                    
#                     # 使用实例计数器来命名文件
#                     self.crop_file_counter += 1
#                     filename = f"{self.crop_file_counter}.jpg"
#                     save_path = os.path.join(save_dir, filename)
#                     cv2.imwrite(save_path, cropped_object)

#     # 【重大修改】修改了 process_video 函数的逻辑
#     def process_video(self, input_path: str, output_path: str, conf_threshold: float = 0.5,
#                       crop_target_class_name: str = None, crop_save_dir: str = None):
#         """
#         处理输入视频，进行目标检测，并将结果保存到输出视频。
#         """
#         if not self.model:
#             print("错误：YOLO模型未初始化，无法处理视频。")
#             return

#         cropping_enabled = False
#         target_class_id = -1
#         if crop_target_class_name and crop_save_dir:
#             if crop_target_class_name in self.class_name_to_id:
#                 cropping_enabled = True
#                 target_class_id = self.class_name_to_id[crop_target_class_name]
#                 print(f"裁剪功能已启用：将裁剪类别 '{crop_target_class_name}' (ID: {target_class_id}) 并保存至 '{crop_save_dir}'")
                
#                 # 在处理开始前，初始化文件计数器
#                 self.crop_file_counter = self.get_highest_file_number(crop_save_dir)
                
#             else:
#                 print(f"警告：指定的裁剪类别 '{crop_target_class_name}' 不存在。将禁用裁剪功能。")
        
#         cap = cv2.VideoCapture(input_path)
#         if not cap.isOpened():
#             print(f"错误: 无法打开输入视频 '{input_path}'")
#             return

#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#         print(f"开始处理视频... 按 'q' 键可提前终止并保存当前进度。")
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             results = self.model(frame, stream=True, conf=conf_threshold, device= '4')
#             overlay = frame.copy()
            
#             for r in results:
#                 if cropping_enabled:
#                     # 不再需要传递帧号
#                     self.crop_and_save_objects(r, frame, target_class_id, crop_save_dir)
                
#                 boxes = r.boxes
#                 for box in boxes:
#                     bbox = box.xyxy[0]
#                     score = float(box.conf[0])
#                     class_id = int(box.cls[0])
#                     self._draw_mmdet_style_box(overlay, bbox, score, class_id)
            
#             alpha = 0.6
#             processed_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
#             out.write(processed_frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("用户手动终止。")
#                 break
        
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()
#         print(f"视频处理完成。结果已保存到 '{output_path}'")


# if __name__ == '__main__':
#     # --- 使用示例 ---
    
#     detector = YoloDetector(model_path='./jg_project/v001-epoch100-aug/weights/best.pt') 

#     # input_video = "./datasets/test-video/720310.mp4" # 确保此路径有效
#     # input_video = "./datasets/test-video/343149.mp4" # 确保此路径有效
    
#     # input_video = "./datasets/test-video/mp4/343149-202507201602-202507201607.mp4" # 确保此路径有效
#     # output_video = "./inference_results/343149-202507201602-202507201607-output.mp4"

#     video_name = '343205-202507201624-202507201629'
#     # 343149-202507201602-202507201607  13
#     # 343205-202507201527-202507201530  1914
#     # 343205-202507201630-202507201639  2888
#     # 343149-202507201745-202507201751  5154
#     # 343205-202507201535-202507201539  6335
#     # 720310-202507201623-202507201635  9551
#     # 343205-202507201520-202507201527  11859
#     # 343205-202507201624-202507201629  12303
    
#     input_video = f"./datasets/test-video/mp4/{video_name}.mp4" # 确保此路径有效
#     output_video = f"./inference_results/{video_name}-output.mp4"




#     # 假设 'yolov8n.pt' 能检测到 'bus'
#     TARGET_CLASS = 'bus'
#     CROP_SAVE_DIRECTORY = f'./inference_results/cropped_{TARGET_CLASS}/'
    
#     if detector.model:
#         detector.process_video(
#             input_path=input_video,
#             output_path=output_video,
#             conf_threshold=0.4,
#             crop_target_class_name=TARGET_CLASS,
#             crop_save_dir=CROP_SAVE_DIRECTORY
#         )










import cv2
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm

class YoloCropper:
    """
    一个专注于从视频源（文件或RTSP流）中检测、裁剪并保存目标物体的工具类。
    【优化版】增加了实时裁剪数量的控制台输出。
    """

    def __init__(self, model_path: str = 'yolov8n.pt'):
        print(f"正在加载YOLO检测模型: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            self.class_name_to_id = {v: k for k, v in self.class_names.items()}
            print(f"✅ 模型加载成功。可检测类别: {list(self.class_names.values())}")
            self.crop_file_counter = 0
        except Exception as e:
            print(f"❌ 错误: 无法加载YOLO模型。请检查路径和安装。错误详情: {e}")
            self.model = None

    def get_highest_file_number(self, directory: str) -> int:
        os.makedirs(directory, exist_ok=True)
        max_num = 0
        print(f"正在扫描目录 '{directory}' 以确定起始编号...")
        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith('.jpg'):
                    try:
                        num = int(os.path.splitext(filename)[0])
                        if num > max_num: max_num = num
                    except ValueError: continue
        except Exception as e: print(f"扫描目录时发生错误: {e}")
        print(f"扫描完成。最大文件编号为: {max_num}。将从下一个编号开始保存。")
        return max_num

    def crop_and_save_objects(self, single_frame_result, original_frame, target_class_id: int, save_dir: str):
        """
        从单帧的检测结果中裁剪出目标并保存，并实时打印计数。
        """
        boxes = single_frame_result.boxes
        for box in boxes:
            if int(box.cls[0]) == target_class_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(original_frame.shape[1], x2), min(original_frame.shape[0], y2)
                
                if x1_c < x2_c and y1_c < y2_c:
                    cropped_object = original_frame[y1_c:y2_c, x1_c:x2_c]
                    
                    self.crop_file_counter += 1
                    filename = f"{self.crop_file_counter}.jpg"
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, cropped_object)
                    
                    # 【核心修改】在控制台打印实时裁剪数量
                    print(f"\n✂️ 已裁剪图片数量: {self.crop_file_counter}", end='\r\n')

    def process_and_crop(self, 
                         source: str, 
                         crop_target_class_name: str, 
                         crop_save_dir: str,
                         conf_threshold: float = 0.5,
                         display_window: bool = True):
        if not self.model:
            print("错误：YOLO模型未初始化，无法处理。")
            return
        
        if crop_target_class_name not in self.class_name_to_id:
            print(f"❌ 错误：要裁剪的目标类别 '{crop_target_class_name}' 不存在于模型的类别列表中。")
            print(f"可用类别为: {list(self.class_names.values())}")
            return
            
        target_class_id = self.class_name_to_id[crop_target_class_name]
        self.crop_file_counter = self.get_highest_file_number(crop_save_dir)
        print(f"✅ 裁剪功能已启用：将裁剪所有 '{crop_target_class_name}' (ID: {target_class_id}) 并保存至 '{crop_save_dir}'")

        is_stream = isinstance(source, str) and source.lower().startswith("rtsp://")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ 错误: 无法打开输入源 '{source}'")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc=f"处理 {os.path.basename(str(source))}", unit="frame", disable=is_stream)

        print(f"🚀 开始处理 {'RTSP流' if is_stream else '视频文件'}... 按 'q' 键可提前终止。")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 【新增】打印换行符，避免最终日志被覆盖
                print() 
                print("\nℹ️ 视频流结束或读取帧失败。")
                break

            results = self.model(frame, stream=True, conf=conf_threshold, device='0', verbose=False)

            for r in results:
                self.crop_and_save_objects(r, frame, target_class_id, crop_save_dir)

            if display_window:
                cv2.imshow('YOLOv8 Cropping Pipeline (Original Stream)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # 【新增】打印换行符
                    print()
                    print("\n用户手动终止。")
                    break
            
            if not is_stream:
                progress_bar.update(1)
        
        cap.release()
        if display_window:
            cv2.destroyAllWindows()
        if not is_stream:
            progress_bar.close()
        
        # 【新增】在流程结束后再次打印最终结果
        print() # 换行
        print(f"处理流程结束。最终共裁剪了 {self.crop_file_counter} 张图片,保存在{crop_save_dir}。")

def run(model_path, 
        INPUT_SOURCE,
        TARGET_CLASS_TO_CROP,
        CROP_SAVE_DIRECTORY= './inference_results/test/bus_crops_from_rtsp', 
        CONFIDENCE_THRESHOLD= 0.45,
        DISPLAY_REALTIME_WINDOW= False):
    # 1. 实例化裁剪器
    cropper = YoloCropper(model_path='./jg_project/v001-epoch100-aug/weights/best.pt')

    # --- 参数配置 ---
    # ... (这部分和您提供的一样，无需修改) ...
    # video_name = '343205-202507201624-202507201629'
    # INPUT_SOURCE = f"./datasets/test-video/mp4/{video_name}.mp4"
    INPUT_SOURCE = f'rtsp://localhost:8556/test-crop'
    
    TARGET_CLASS_TO_CROP = 'bus'
    CROP_SAVE_DIRECTORY = f'./inference_results/test/bus_crops_from_rtsp/'
    CONFIDENCE_THRESHOLD = 0.45
    
    DISPLAY_REALTIME_WINDOW = False # 设为False可在服务器上无界面运行

    # --- 运行处理 ---
    if cropper.model:
        cropper.process_and_crop(
            source=INPUT_SOURCE,
            crop_target_class_name=TARGET_CLASS_TO_CROP,
            crop_save_dir=CROP_SAVE_DIRECTORY,
            conf_threshold=CONFIDENCE_THRESHOLD,
            display_window=DISPLAY_REALTIME_WINDOW
        )
if __name__ == '__main__':
    # 1. 实例化裁剪器
    cropper = YoloCropper(model_path='./jg_project/v001-epoch100-aug/weights/best.pt')

    # --- 参数配置 ---
    # ... (这部分和您提供的一样，无需修改) ...
    # video_name = '343205-202507201624-202507201629'
    # INPUT_SOURCE = f"./datasets/test-video/mp4/{video_name}.mp4"
    INPUT_SOURCE = f'rtsp://localhost:8556/test-crop'
    
    TARGET_CLASS_TO_CROP = 'bus'
    CROP_SAVE_DIRECTORY = f'./inference_results/test/bus_crops_from_rtsp/'
    CONFIDENCE_THRESHOLD = 0.45
    
    DISPLAY_REALTIME_WINDOW = False # 设为False可在服务器上无界面运行

    # --- 运行处理 ---
    if cropper.model:
        cropper.process_and_crop(
            source=INPUT_SOURCE,
            crop_target_class_name=TARGET_CLASS_TO_CROP,
            crop_save_dir=CROP_SAVE_DIRECTORY,
            conf_threshold=CONFIDENCE_THRESHOLD,
            display_window=DISPLAY_REALTIME_WINDOW
        )