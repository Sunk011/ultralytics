"""
使用yolo模型推理,并将检测到的bus保存在指定目录
"""


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