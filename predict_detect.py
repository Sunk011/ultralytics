import os
import datetime
import cv2
import numpy as np
from torch._functorch.config import activation_memory_budget
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results
from utils.input_handler import InputStreamHandler
from utils.config_loader import Config
from utils.draw_reticle_box import draw_reticle_box
from utils.draw_attribute_panel import draw_floating_panel_with_arrow

class Yolodetect:
    """
    一个使用YOLO模型进行检测、裁剪和可视化的工具类。
    【优化版】使用MMDetection风格进行绘制。
    """
    def __init__(self, model_path: str = 'yolov8n.pt',  output_path: str = './output'):
        print(f"正在加载YOLO检测模型: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            self.class_name_to_id = {v: k for k, v in self.class_names.items()}
            print(f"✅ 模型加载成功。可检测类别: {list(self.class_names.values())}")
            
            np.random.seed(41)
            self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
            self.output_path = output_path
            self.output_path_crop = os.path.join(self.output_path, 'crops')
            self.output_path_draw = os.path.join(self.output_path, 'draws')
            self.output_path_video = os.path.join(self.output_path, 'videos')
            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.output_path_crop, exist_ok=True)
            os.makedirs(self.output_path_draw, exist_ok=True)
            os.makedirs(self.output_path_video, exist_ok=True)
            
            self.croped_counter = self._get_dir_file_number(self.output_path_crop)
            self.drawed_counter = self._get_dir_file_number(self.output_path_draw)

            # 保存视频相关
            self.video_writer = None
            self.target_frames = 0
            self.video_save_path = ''
        except Exception as e:
            print(f"❌ 错误: 无法加载YOLO模型。请检查路径和安装。错误详情: {e}")
            self.model = None

    def handle_crop_task(self, result, image, target_class_id):
        """处理裁剪任务"""
        cropped_image, status = self.crop_target(result, image, target_class_id)
        if status:
            self.save_result(cropped_image, task='crop_image')

    def handle_draw_task(self, result, image, target_class_id):
        """处理绘制任务"""
        drawed_image, status = self.draw_target(result, image, target_class_id)
        if status:
            self.save_result(drawed_image, task='draw_image')

    def handle_video_task(self, result, image, target_class_id):
        """处理视频保存任务"""
        drawed_image, _ = self.draw_target(result, image, target_class_id)
        self.save_result(drawed_image, task='save_video', fps=25)
    
    def handle_display_info(self, result, target_class_id):
        """处理显示信息"""
        # TODO

    def _get_dir_file_number(self, directory: str) -> int:
        os.makedirs(directory, exist_ok=True)
        max_num = 0
        print(f"正在扫描目录 '{directory}' 以确定起始编号...")
        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    try:
                        # 兼容 crop_1.jpg 和 1.jpg 两种格式
                        num_str = os.path.splitext(filename)[0]
                        if '_' in num_str:
                            num_str = num_str.split('_')[-1]
                        num = int(num_str)
                        if num > max_num: max_num = num
                    except ValueError: continue
        except Exception as e: print(f"扫描目录时发生错误: {e}")
        print(f"扫描完成。最大文件编号为: {max_num}。")
        return max_num

    def infer(self, frame: np.ndarray, stream: bool = True, conf_threshold: float = 0.5, device: str = '0', verbose: bool = False) -> list[Results]:
        results = self.model(frame, stream=stream, conf=conf_threshold, device=device, verbose=verbose)
        return results
    
    def crop_target(self, det_result: Results, image: np.ndarray, target_class_id: list) -> tuple[np.ndarray, bool]:
        """
        根据单帧的YOLO推理结果，从原始图像中裁剪出指定类别的目标图像。若找到符合条件的目标则返回裁剪后的图像，
        若未找到则返回原图。同时返回一个布尔值表示是否成功裁剪。

        Args:
            det_result (Results): 单帧的YOLO推理结果对象
            image (np.ndarray): 原始的OpenCV图像帧(BGR格式)
            target_class_id (list): 要裁剪的目标类别ID列表。-1代表所有类别

        Returns:
            tuple[np.ndarray, bool]: 元组，第一个元素为裁剪后的图像，第二个元素表示是否成功裁剪到目标
        """
        boxes = det_result.boxes
        status = False
        for box in boxes:
            class_id = int(box.cls[0])
            if  (-1 in target_class_id) or (class_id in target_class_id):
                status = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(image.shape[1], x2), min(image.shape[0], y2)
                
                if x1_c < x2_c and y1_c < y2_c:
                    cropped_object = image[y1_c:y2_c, x1_c:x2_c]
                    return cropped_object, status
        return image, status
    
    def draw_target(self, det_result: Results, image: np.ndarray, target_class_id: list = [-1]) -> tuple[np.ndarray, bool]:

        """
        在图像上绘制指定类别的目标边界框,同时返回一个布尔值表示是否成功绘制目标。

        Args:
            det_result (Results): 单帧的YOLO推理结果对象。
            image (np.ndarray): 原始的OpenCV图像帧(BGR格式)。
            target_class_id (list): 要绘制的目标类别ID列表，-1代表所有类别。

        Returns:
            tuple[np.ndarray, bool]: 元组，第一个元素为绘制处理后的图像帧，第二个元素表示是否成功绘制目标。
        """
        # 创建一个用于绘制的覆盖层，与原图大小相同
        overlay = image.copy()
        final_image = image.copy() # 用于最终混合
        boxes = det_result.boxes
        status = False
        
        
        did_draw_anything = False
        for box in boxes:
            class_id = int(box.cls[0])
            should_draw = (-1 in target_class_id) or (class_id in target_class_id)
            
            if should_draw:
                status = True
                did_draw_anything = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # score = float(box.conf[0])  # 置信度
                color = self.colors[class_id].tolist()
                # label = f'{class_id}:{self.class_names.get(class_id, f"ID_{class_id}")} {score:.2f}'
                # label = f'{self.class_names.get(class_id, f"ID_{class_id}")} {score:.2f}'
                # print(f"绘制目标: {label} 位置: ({x1}, {y1}) -> ({x2}, {y2})")
                
                # --- 核心绘制逻辑 ---
                # 1. 在覆盖层上绘制半透明的填充矩形
                alpha = 0.2  # 透明度
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # # 2. 在最终图像上绘制不透明的边界框线条 绘制检测框:矩形
                # cv2.rectangle(final_image, (x1, y1), (x2, y2), color, 2) # 线条粗细为2
                # 2.绘制类似于瞄准框的检测框
                final_image = draw_reticle_box(final_image, [x1, y1, x2, y2], percent=0.2, color=color)
                
                # 3. 计算文本大小并绘制标签背景
                # font_scale = 1 # 调小字体
                # font_thickness = 2
                # (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                # 确保标签背景不会超出图像顶部
                # label_y1 = max(y1, text_h + 10) # 标签放在框的左上角内侧或外侧
                # cv2.rectangle(final_image, (x1, label_y1 - text_h - baseline), (x1 + text_w, label_y1), color, -1)
                
                # 4. 绘制文本
                # cv2.putText(final_image, label, (x1, label_y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # 如果进行了绘制，则将覆盖层与最终图像混合
        if did_draw_anything:
            final_image = cv2.addWeighted(overlay, alpha, final_image, 1 - alpha, 0)
        

        return final_image, status
    
    def save_result(self, posted_image: np.ndarray, task: str ='save_images', fps: int = 25) -> str:
        """
        保存绘制后的图像到指定路径。

        Args:
            posted_image (np.ndarray): 绘制后的图像帧。
            task (str): 任务类型，默认'save_images',可选任务类型：'crop_image','save_image', 'save_video'。

        Returns:
            str: 成功保存的文件路径，或空字符串表示保存失败。
        """
        
        # --- 保存 ---
        if task == 'save_video':
            # 初始化视频写入器（如果尚未初始化）
            if not hasattr(self, 'video_writer') or self.video_writer is None:
                # 使用时间戳生成唯一文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_save_path = os.path.join(self.output_path_video, f"video_{timestamp}.mp4")
                self.video_save_path = video_save_path
                
                # 获取图像尺寸和设置视频参数
                height, width = posted_image.shape[:2]
                
                # 初始化视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))
                self.video_frame_count = 0
                print(f"start video: {video_save_path}")
            
            try:
                # 写入当前帧
                self.video_writer.write(posted_image)
                self.video_frame_count += 1
            except Exception as e:
                print(f"❌ 视频写入错误: {e}")
                self.video_writer.release()
                self.video_writer = None
                return ""
            
            # 检查是否达到目标录制时长
            if self.video_frame_count >= (fps * 60):
                self.video_writer.release()
                print(f"✅ 视频已成功保存到: {self.output_path_video}")
                # 重置视频写入器状态，准备下一次录制
                self.video_writer = None
                # self.is_recording = False  # 更新录制状态
                return self.video_save_path
            
            # 仍在录制中，返回当前进度
            return f"视频录制中: {self.video_frame_count}/{self.target_frames} 帧"
        else:
            if task == 'crop_image':
                save_path = self.output_path_crop
                file_num = self.croped_counter
                self.croped_counter += 1
            else:
                save_path = self.output_path_draw
                file_num = self.drawed_counter
                self.drawed_counter += 1
            try:
                filename = f"{task}_{file_num}.jpg"
                save_path = os.path.join(save_path, filename)
                
                print(f"正在保存图片到: {save_path}")
                cv2.imwrite(save_path, posted_image)
                print(f"✅ 结果已成功保存到: {self.output_path}, 文件名: {filename}") 
                return save_path
            except Exception as e:
                print(f"❌ 保存图片时发生错误到 '{self.output_path}': {e}")
                return ""
            


def start(config_path: str = 'config.yaml'):
    # 加载配置文件
    config = Config(config_path)
    
    # 从配置中获取参数
    MODEL_PATH = config.get('model.path')
    INPUT_SOURCE = config.get('input.source')
    OUTPUT_PATH = config.get('output.path', './output')
    TARGET_CLASS_ID = config.get('target_class_id', default=[-1])
    DEVICE = config.get('infer_setting.device', '0')
    CONF = config.get('infer_setting.confidence_threshold', default=0.5)
    TASK_TYPES = config.get('task.type', default='save_video')

    TASK_TYPES= TASK_TYPES.split(',') if hasattr(config, 'task') and hasattr(config.task, 'type') else []
    
    print(TASK_TYPES)
    
    print("=====Start =====")
    from queue import Queue
    
    yolo_det = Yolodetect(MODEL_PATH,OUTPUT_PATH)
    
    print(f"目标类别ID: {TARGET_CLASS_ID}")
    
    # 任务映射字典：将任务类型映射到对应的处理函数
    task_handlers = {
        'crop_image': yolo_det.handle_crop_task,
        'draw_image': yolo_det.handle_draw_task,
        'save_video': yolo_det.handle_video_task
    }
    # 根据配置初始化要执行的任务列表
    active_tasks = [task_handlers[task.strip()] for task in TASK_TYPES if task.strip() in task_handlers]
    # print(active_tasks)
    # return ''
    
    camera = InputStreamHandler(INPUT_SOURCE, Queue())
    camera.run()
    
    print(' ---------- Process Thread Start ---------- ')
    
    frame_count = 0
    try:
        while True:
            image = camera.frame_queue.get()
            progress_bar = tqdm(desc=f"处理RTSP流", unit="frame")  # 不设置total，动态显示已处理帧数
            progress_bar.update(frame_count)
            
            frame_count += 1
            
            # 1. 推理
            det_results = yolo_det.infer(image, stream=True, conf_threshold=CONF, device=DEVICE)
            
            for result in det_results:
                # 执行所有激活的任务
                for task_handler in active_tasks:
                    task_handler(result, image, TARGET_CLASS_ID)
    except KeyboardInterrupt:
        print("\n =CTRL= KeyboardInterrupt! done ============ ")
    except Exception as e:
        print(f"\n =ERROR= Exception! done ======Reason: {e}")
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    print('\n =^_^= All done, See you ~')

if __name__ == '__main__':
    start(r'./ultralytics/config_yaml/config_detect.yaml')