# import os
# import cv2
# from ultralytics import YOLO
# from tqdm import tqdm
# import time
# from datetime import datetime

# class TwoStagePipeline:
#     """
#     一个实现两阶段检测流程的管道。
#     支持图片、视频文件和RTSP流的推理，并提供灵活的保存选项。
#     """
#     def __init__(self, detection_model_path: str, classification_model_path: str):
#         print("正在加载模型...")
#         try:
#             self.detector = YOLO(detection_model_path)
#             self.classifier = YOLO(classification_model_path)
#             self.classifier_names = self.classifier.names
#             print("✅ 所有模型加载成功。")
#             print(f"检测器类别: {list(self.detector.names.values())[:5]}...") # 显示部分检测类别
#             print(f"分类器类别: {list(self.classifier_names.values())}")
#         except Exception as e:
#             raise IOError(f"加载模型时出错: {e}")

#     def _draw_result(self, image, box, label: str, confidence: float):
#         """在图片上绘制单个结果框和标签的辅助函数。"""
#         x1, y1, x2, y2 = map(int, box)
#         is_target = "positive" in label.lower() or "target" in label.lower() # 假设正样本文件夹名为 positive_samples
#         color = (0, 220, 0) if is_target else (0, 0, 220)
        
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#         display_text = f"{label} ({confidence:.2f})"
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.8
#         font_thickness = 2
#         (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
#         cv2.rectangle(image, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
#         cv2.putText(image, display_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

#     def _process_frame(self, frame, target_detection_class: str = 'bus', conf_threshold: float = 0.4):
#         """处理单帧图像的内部核心逻辑。"""
#         # --- 阶段一：检测 ---
#         print(f"正在检测目标: {target_detection_class} (置信度阈值: {conf_threshold})", end='\r')
#         det_results = self.detector(frame, conf=conf_threshold, classes=[list(self.detector.names.values()).index(target_detection_class)], verbose=False)
        
        
#         # --- 阶段二：分类 ---
#         for box in det_results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
#             if x1 >= x2 or y1 >= y2: continue
                
#             cropped_img = frame[y1:y2, x1:x2]
#             cls_results = self.classifier(cropped_img, verbose=False)
            
#             top1_index = cls_results[0].probs.top1
#             top1_confidence = cls_results[0].probs.top1conf
#             predicted_class_name = self.classifier_names[top1_index]
            
#             self._draw_result(frame, box.xyxy[0], predicted_class_name, top1_confidence)
        
#         return frame

#     def _process_video_or_stream(self, source, output_dir: str, save_results: bool, rtsp_segment_time: int = 60):
#         """处理视频文件或实时流的内部方法。"""
#         cap = cv2.VideoCapture(source)
#         if not cap.isOpened():
#             print(f"错误: 无法打开视频源 '{source}'")
#             return

#         is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")
        
#         video_writer = None
#         segment_start_time = time.time()
        
#         # 如果是普通视频文件，预先设置好VideoWriter
#         if save_results and not is_rtsp:
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             filename = f"output_{os.path.basename(source)}"
#             output_path = os.path.join(output_dir, filename)
#             video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#             print(f"将保存处理后的视频到: {output_path}")

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("视频流结束或读取帧失败。")
#                 break
            
#             # --- RTSP分段保存逻辑 ---
#             if save_results and is_rtsp:
#                 current_time = time.time()
#                 # 如果当前分段已达到时长，则关闭当前writer
#                 if video_writer and (current_time - segment_start_time >= rtsp_segment_time):
#                     video_writer.release()
#                     video_writer = None
#                     print(f"RTSP已分段。")
                
#                 # 如果没有writer，则创建一个新的
#                 if not video_writer:
#                     segment_start_time = current_time
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = f"rtsp_segment_{timestamp}.mp4"
#                     output_path = os.path.join(output_dir, filename)
                    
#                     fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25 # 默认25fps
#                     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                     video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#                     print(f"开始录制新的视频分段: {output_path}")

#             processed_frame = self._process_frame(frame.copy())

#             # 如果需要保存，则写入帧
#             if save_results and video_writer:
#                 video_writer.write(processed_frame)

#             # 实时显示结果
#             # cv2.imshow("Two-Stage Pipeline", processed_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("用户手动终止。")
#                 break
        
#         # 释放资源
#         if video_writer:
#             video_writer.release()
#         cap.release()
#         cv2.destroyAllWindows()

#     def process(self, source, save_results: bool = False, output_image_dir: str = 'outputs/images', output_video_dir: str = 'outputs/videos', output_rtsp_dir: str = 'outputs/rtsp_segments', rtsp_segment_time: int = 60):
#         """
#         统一的处理入口，自动识别输入源类型。

#         Args:
#             source (str or int): 输入源。可以是图片路径, 视频路径, RTSP URL, 或摄像头索引(0, 1, ...).
#             save_results (bool): 是否保存处理结果。
#             output_image_dir (str): 保存处理后图片的目录。
#             output_video_dir (str): 保存处理后视频的目录。
#             output_rtsp_dir (str): 保存RTSP流分段的目录。
#             rtsp_segment_time (int): RTSP流分段保存的时长（秒）。
#         """
#         # 创建所有输出目录
#         os.makedirs(output_image_dir, exist_ok=True)
#         os.makedirs(output_video_dir, exist_ok=True)
#         os.makedirs(output_rtsp_dir, exist_ok=True)
        
#         # --- 判断输入源类型 ---
#         source_str = str(source).lower()
#         if source_str.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#             # 处理单张图片
#             filename = os.path.basename(source)
#             output_path = os.path.join(output_image_dir, f"output_{filename}")
#             frame = cv2.imread(source)
#             frame = frame.copy()
#             if frame is None:
#                 print(f"错误: 无法读取图片 '{source}'")
#                 return
#             processed_frame = self._process_frame(frame)
#             if save_results:
#                 cv2.imwrite(output_path, processed_frame)
#                 print(f"处理完成！结果已保存至: {output_path}")
#             # cv2.imshow("Two-Stage Pipeline", processed_frame)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
            
#         elif source_str.startswith("rtsp://"):
#             # 处理RTSP流
#             print("检测到RTSP流，开始处理...")
#             self._process_video_or_stream(source, output_rtsp_dir, save_results, rtsp_segment_time)
        
#         else:
#             # 处理视频文件或摄像头
#             print("检测到视频文件或摄像头，开始处理...")
#             self._process_video_or_stream(source, output_video_dir, save_results)


# # --- 使用示例 ---
# if __name__ == '__main__':
#     # 1. 设置模型路径
#     DETECTION_MODEL_PATH = r'./jg_project/v001-epoch100-aug/weights/best.pt'
#     CLASSIFICATION_MODEL_PATH = r'./jg_project/cls/test3/weights/best.pt' # <--- 修改这里

#     # 2. 设置保存选项和目录
#     SAVE_RESULTS = True  # True: 保存所有结果, False: 只实时显示不保存
    
#     OUTPUT_IMAGE_DIR = 'inference_outputs/test/images'
#     OUTPUT_VIDEO_DIR = 'inference_outputs/test/videos'
#     OUTPUT_RTSP_DIR = 'inference_outputs/test/rtsp_segments'
#     RTSP_SEGMENT_DURATION_SECONDS = 90 # 1分钟

#     # 3. 创建管道实例
#     try:
#         pipeline = TwoStagePipeline(
#             detection_model_path=DETECTION_MODEL_PATH,
#             classification_model_path=CLASSIFICATION_MODEL_PATH
#         )
        
#         # --- 根据你的需求，选择一种输入源来运行 ---

#         # === 示例1: 处理单张图片 ===
#         # INPUT_SOURCE = 'path/to/your/street_view.jpg' # <--- 修改这里

#         # === 示例2: 处理本地视频文件 ===
#         INPUT_SOURCE = '/home/sk/project/datasets/test-video/mp4/343149-202507201602-202507201607.mp4' # <--- 修改这里

#         # === 示例3: 处理RTSP视频流 ===
#         # INPUT_SOURCE = 'rtsp://username:password@ip_address:port/stream_path' # <--- 修改这里
        
#         # 4. 运行管道
#         pipeline.process(
#             source=INPUT_SOURCE,
#             save_results=SAVE_RESULTS,
#             output_image_dir=OUTPUT_IMAGE_DIR,
#             output_video_dir=OUTPUT_VIDEO_DIR,
#             output_rtsp_dir=OUTPUT_RTSP_DIR,
#             rtsp_segment_time=RTSP_SEGMENT_DURATION_SECONDS
#         )

#     except Exception as e:
#         print(f"程序运行出错: {e}")
        
        








import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm # 引入tqdm进度条库
import time
from datetime import datetime

class TwoStagePipeline:
    """
    一个实现两阶段检测流程的管道。
    支持图片、视频文件和RTSP流的推理，并提供灵活的保存选项。
    【优化版】
    """
    def __init__(self, detection_model_path: str, classification_model_path: str):
        print("正在加载模型...")
        try:
            self.detector = YOLO(detection_model_path)
            self.classifier = YOLO(classification_model_path)
            self.classifier_names = self.classifier.names
            print("✅ 所有模型加载成功。")
            print(f"检测器类别: {list(self.detector.names.values())[:5]}...")
            print(f"分类器类别: {list(self.classifier_names.values())}")
        except Exception as e:
            raise IOError(f"加载模型时出错: {e}")

    def _draw_result(self, image, box, label: str, confidence: float):
        """在图片上绘制单个结果框和标签的辅助函数。"""
        x1, y1, x2, y2 = map(int, box)
        # 【优化】假设正样本类别名为 'target_bus'
        is_target = "target_bus" in label.lower()
        color = (0, 220, 0) if is_target else (0, 0, 220)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        display_text = f"{label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(image, display_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

    def _process_frame(self, frame, target_detection_class: str, conf_threshold: float):
        """处理单帧图像的内部核心逻辑。"""
        # --- 阶段一：检测 ---
        det_results = self.detector(frame, conf=conf_threshold, classes=[list(self.detector.names.values()).index(target_detection_class)], verbose=False)
        
        # print(f"\ndet_resutls:\n{det_results}")
        
        # --- 阶段二：分类 ---
        for box in det_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x1 >= x2 or y1 >= y2: continue
                
            cropped_img = frame[y1:y2, x1:x2]
            cls_results = self.classifier(cropped_img, verbose=False)
            
            top1_index = cls_results[0].probs.top1
            top1_confidence = cls_results[0].probs.top1conf
            predicted_class_name = self.classifier_names[top1_index]
            
            # 仅绘制分类为target_bus的目标
            if predicted_class_name == 'target_bus':
                self._draw_result(frame, box.xyxy[0], predicted_class_name, top1_confidence)
        
        return frame

    def _process_video_or_stream(self, source, output_dir: str, save_results: bool, target_detection_class: str, conf_threshold: float, rtsp_segment_time: int):
        """处理视频文件或实时流的内部方法。"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ 错误: 无法打开视频源 '{source}'。请检查路径或RTSP URL是否正确，以及文件是否损坏或缺少解码器。")
            return

        is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 视频信息: {width}x{height}, {fps:.2f} FPS, 共 {total_frames if not is_rtsp else 'N/A'} 帧。")

        video_writer = None
        # segment_start_time = time.time()
        
        if save_results and not is_rtsp:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            base_name = os.path.basename(source)
            filename = f"output_{os.path.splitext(base_name)[0]}.mp4"
            output_path = os.path.join(output_dir, filename)
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 将保存处理后的视频到: {output_path}")

        # 为本地视频添加进度条
        progress_bar = tqdm(total=total_frames, desc="处理视频中", unit="frame", disable=is_rtsp)

        while True:
            ret, frame = cap.read()
            if not ret:
                if not is_rtsp: print("\n✅ 视频处理完成。") 
                else: print("ℹ️ 视频流结束或读取帧失败。")
                break
            
            # # RTSP分段保存逻辑
            # if save_results and is_rtsp:
            #     segment_start_time = current_time
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     filename = f"rtsp_segment_{timestamp}.mp4"
            #     output_path = os.path.join(output_dir, filename)
                
            #     fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25 # 默认25fps
            #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #     video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            #     print(f"开始录制新的视频分段: {output_path}")

            # 【修复】将参数正确传递给_process_frame
            processed_frame = self._process_frame(frame.copy(), target_detection_class, conf_threshold)

            if save_results and video_writer:
                video_writer.write(processed_frame)

            # cv2.imshow("Two-Stage Pipeline", processed_frame) # 服务器运行时注释掉
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户手动终止。")
                break
            
            if not is_rtsp:
                progress_bar.update(1)
        
        if video_writer: video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        if not is_rtsp: progress_bar.close()

    def process(self, source, save_results: bool = False, output_image_dir: str = 'outputs/images', output_video_dir: str = 'outputs/videos', output_rtsp_dir: str = 'outputs/rtsp_segments', rtsp_segment_time: int = 60, target_detection_class: str = 'bus', conf_threshold: float = 0.4):
        """统一的处理入口，自动识别输入源类型。"""
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_video_dir, exist_ok=True)
        os.makedirs(output_rtsp_dir, exist_ok=True)
        
        source_str = str(source).lower()
        if source_str.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            frame = cv2.imread(source)
            if frame is None:
                print(f"❌ 错误: 无法读取图片 '{source}'")
                return
            
            # 【修复】将参数正确传递给_process_frame
            processed_frame = self._process_frame(frame.copy(), target_detection_class, conf_threshold)
            
            if save_results:
                filename = os.path.basename(source)
                output_path = os.path.join(output_image_dir, f"output_{filename}")
                cv2.imwrite(output_path, processed_frame)
                print(f"处理完成！结果已保存至: {output_path}")
            
            cv2.imshow("Two-Stage Pipeline", processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        elif source_str.startswith("rtsp://"):
            print("检测到RTSP流，开始处理...")
            # 【修复】将参数正确传递
            self._process_video_or_stream(source, output_rtsp_dir, save_results, target_detection_class, conf_threshold, rtsp_segment_time)
        
        else:
            print("检测到视频文件或摄像头，开始处理...")
            # 【修复】将参数正确传递
            self._process_video_or_stream(source, output_video_dir, save_results, target_detection_class, conf_threshold, rtsp_segment_time)

# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 设置模型路径
    DETECTION_MODEL_PATH = r'./jg_project/v001-epoch100-aug/weights/best.pt'
    CLASSIFICATION_MODEL_PATH = r'./jg_project/cls/test3/weights/best.pt'

    # 2. 设置推理参数
    SAVE_RESULTS = True
    TARGET_CLASS = 'bus' # 确保你的检测器能识别这个类别
    CONF_THRESHOLD = 0.45

    # 3. 设置输出目录
    OUTPUT_IMAGE_DIR = 'inference_outputs/test/images'
    OUTPUT_VIDEO_DIR = 'inference_outputs/test/videos'
    OUTPUT_RTSP_DIR = 'inference_outputs/test/rtsp_segments'
    RTSP_SEGMENT_DURATION_SECONDS = 90

    # 4. 创建管道实例
    try:
        pipeline = TwoStagePipeline(
            detection_model_path=DETECTION_MODEL_PATH,
            classification_model_path=CLASSIFICATION_MODEL_PATH
        )
        
        # --- 选择一种输入源 ---
        # INPUT_SOURCE = 'path/to/image.jpg'
        INPUT_SOURCE = '/home/sk/project/datasets/test-video/mp4/343149-202507201602-202507201607.mp4'
        # INPUT_SOURCE = 'rtsp://...'
        
        # 5. 运行管道
        pipeline.process(
            source=INPUT_SOURCE,
            save_results=SAVE_RESULTS,
            output_image_dir=OUTPUT_IMAGE_DIR,
            output_video_dir=OUTPUT_VIDEO_DIR,
            output_rtsp_dir=OUTPUT_RTSP_DIR,
            rtsp_segment_time=RTSP_SEGMENT_DURATION_SECONDS,
            target_detection_class=TARGET_CLASS,
            conf_threshold=CONF_THRESHOLD
        )

    except Exception as e:
        print(f"程序运行出错: {e}")