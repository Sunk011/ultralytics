# import os
# import cv2
# import numpy as np
# import torch
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction

# class SahiProcessor:
#     """
#     一个使用YOLO和SAHI进行切片推理、裁剪和可视化的工具类。
#     专为与多线程视频流处理InputStreamHandler集成而设计。
#     """
#     def __init__(self, model_path: str, output_path: str = './output'):
#         """
#         初始化处理器,使用SAHI加载YOLO模型。

#         Args:
#             model_path (str): 您自定义的YOLO模型的.pt文件路径。
#             output_path (str): 输出结果的保存目录。
#         """
#         print(f"正在使用SAHI加载YOLO模型: {model_path}")
#         try:
#             # 检查设备
#             if torch.cuda.is_available():
#                 self.device = torch.device("cuda:0")
#                 print(f"✅ 检测到CUDA，将使用设备: {self.device}")
#             else:
#                 self.device = torch.device("cpu")
#                 print("⚠️ 未检测到CUDA，将使用CPU。")

#             # 使用SAHI的标准方式加载模型
#             self.detection_model = AutoDetectionModel.from_pretrained(
#                 model_type='yolov8',
#                 model_path=model_path,
#                 confidence_threshold=0.3, # 初始阈值，可在推理时覆盖
#                 device=self.device,
#             )
#             self.class_names = self.detection_model.model.names
#             self.class_name_to_id = {v: k for k, v in self.class_names.items()}
#             print(f"✅ 模型加载成功。可检测类别: {list(self.class_names.values())}")
            
#             self.output_path = output_path
#             self.file_counter = self._get_dir_file_number(self.output_path)

#         except Exception as e:
#             print(f"❌ 错误: 无法加载模型。错误详情: {e}")
#             self.detection_model = None

#     def _get_dir_file_number(self, directory: str) -> int:
#         os.makedirs(directory, exist_ok=True)
#         max_num = 0
#         print(f"正在扫描目录 '{directory}' 以确定起始编号...")
#         try:
#             for filename in os.listdir(directory):
#                 if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#                     try:
#                         num_str = os.path.splitext(filename)[0]
#                         # 兼容 crop_1.jpg 这样的格式
#                         if num_str.startswith("crop_"):
#                             num_str = num_str[5:]
#                         num = int(num_str)
#                         if num > max_num: max_num = num
#                     except ValueError: continue
#         except Exception as e: print(f"扫描目录时发生错误: {e}")
#         print(f"扫描完成。最大文件编号为: {max_num}。")
#         return max_num

#     def infer_with_sahi(self, 
#                         frame: np.ndarray, 
#                         slice_size: int = 640, 
#                         overlap_ratio: float = 0.2, 
#                         nms_threshold: float = 0.5,
#                         conf_threshold: float = 0.5) -> list[dict]:
#         """
#         对输入的图像帧执行SAHI切片推理。

#         Args:
#             frame (np.ndarray): 输入的图像帧 (OpenCV BGR格式)。
#             slice_size (int): 切片的高度和宽度。
#             overlap_ratio (float): 切片之间的重叠率。
#             conf_threshold (float): 用于过滤最终结果的置信度阈值。

#         Returns:
#             list[dict]: 一个包含所有检测结果的列表，每个结果是一个字典。
#                         e.g., [{'box': [x1,y1,x2,y2], 'score': 0.9, 'class_id': 1, 'class_name': 'bus'}, ...]
#         """
#         if not self.detection_model: return []

#         sahi_result = get_sliced_prediction(
#                 frame,
#                 self.detection_model,
#                 slice_height=slice_size,
#                 slice_width=slice_size,
#                 overlap_height_ratio=overlap_ratio,
#                 overlap_width_ratio=overlap_ratio,
#                 postprocess_match_threshold=nms_threshold, # 进行后处理的阈值
#                 verbose= 2
#                 # confidence_threshold=conf_threshold # 置信度阈值（修正参数名）
#             )
        
#         # 将SAHI结果转换为简单的字典列表
#         predictions_list = []
#         for pred in sahi_result.object_prediction_list:
#             predictions_list.append({
#                 "box": [int(coord) for coord in pred.bbox.to_xyxy()],
#                 "score": pred.score.value,
#                 "class_id": pred.category.id,
#                 "class_name": pred.category.name
#             })
#         return predictions_list
    
#     def crop_target(self, predictions: list[dict], image: np.ndarray, target_class_id: int) -> None:
#         """
#         从预测结果列表中裁剪出指定类别的目标并保存。

#         Args:
#             predictions (list[dict]): 来自 infer_with_sahi 方法的预测结果列表。
#             image (np.ndarray): 原始的、未被修改的图像帧。
#             target_class_id (int): 要裁剪的目标的类别ID。
#         """
#         image_copy = image.copy()
#         for pred in predictions:
#             if pred['class_id'] == target_class_id:
#                 x1, y1, x2, y2 = pred['box']
#                 x1_c, y1_c = max(0, x1), max(0, y1)
#                 x2_c, y2_c = min(image_copy.shape[1], x2), min(image_copy.shape[0], y2)
                
#                 if x1_c < x2_c and y1_c < y2_c:
#                     cropped_object = image_copy[y1_c:y2_c, x1_c:x2_c]
#                     self.file_counter += 1
#                     filename = f"crop_{self.file_counter}.jpg"
#                     save_path = os.path.join(self.output_path, filename)
#                     cv2.imwrite(save_path, cropped_object)
#                     print(f"✂️ 已裁剪图片数量: {self.file_counter}", end='\r')


# import os
# from queue import Queue
# from tqdm import tqdm
# from utils.input_handler import InputStreamHandler

# if __name__ == '__main__':
#     # 1. --- 参数配置 ---
#     MODEL_PATH = r'./jg_project/v001-epoch100-aug/weights/best.pt'
#     INPUT_SOURCE = r'./datasets/test-video/mp4/343205-2025v07201624-202507201629.mp4'
#     # INPUT_SOURCE = 'rtsp://your_rtsp_url'
#     OUTPUT_PATH = r'./output_sahi_crops'

#     # --- SAHI 和 推理参数 ---
#     SLICE_SIZE = 640
#     OVERLAP_RATIO = 0.2
#     NMS_THRESHOLD = 0.5
#     CONF_THRESHOLD = 0.45
#     DEVICE = '0' # SAHI会自动处理设备，这里保留给未来可能的扩展

#     # --- 裁剪目标参数 ---
#     # 【注意】这里使用类别名称，更友好。如果模型中没有该类别，程序会报错并退出。
#     TARGET_CLASS_NAME = 'bus'
    
#     print("===== 开始初始化 =====")
    
#     # 2. --- 初始化处理器和视频流 ---
#     # 初始化SAHI处理器
#     sahi_proc = SahiProcessor(MODEL_PATH, OUTPUT_PATH)
    
#     # 检查目标类别是否存在
#     if TARGET_CLASS_NAME not in sahi_proc.class_name_to_id:
#         print(f"❌ 错误: 目标类别 '{TARGET_CLASS_NAME}' 不存在于模型中。")
#         print(f"可用类别: {list(sahi_proc.class_names.values())}")
#         exit()
#     target_class_id = sahi_proc.class_name_to_id[TARGET_CLASS_NAME]

#     print('---------- start ----------')
#     # 初始化视频流处理器
#     frame_queue = Queue()
#     camera = InputStreamHandler(INPUT_SOURCE, frame_queue)
#     camera.run()
    
#     print(' ---------- SAHI Process Thread Start ---------- ')
#     frame_count = 0
#     try:
#         while True:
#             # 从队列中获取原始帧
#             image = frame_queue.get()
#             # if image is None: # 可以约定一个None作为结束信号
#             #     break
#             frame_count += 1
            
#             # --- 核心处理步骤 ---
#             # 1. 使用SAHI进行切片推理，获取预测结果列表
#             predictions = sahi_proc.infer_with_sahi(
#                 frame=image, 
#                 slice_size=SLICE_SIZE,
#                 overlap_ratio=OVERLAP_RATIO,
#                 nms_threshold = NMS_THRESHOLD,
#               # conf_threshold=CONF_THRESHOLD
#             )
            
#             # 2. 如果有预测结果，则调用裁剪函数
#             if predictions:
#                 sahi_proc.crop_target(
#                     predictions=predictions, 
#                     image=image, 
#                     target_class_id=target_class_id
#                 )

#     except KeyboardInterrupt:
#         print("\n =CTRL= 检测到键盘中断! 正在退出...")
#     except Exception as e:
#         print(f"\n =ERROR= 发生异常! 原因: {e}")
#     finally:
#         # 确保线程被安全停止
#         camera.stop()
#     print(' =^_^= All done, See you ~')


                    






import os
import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class SahiProcessor:
    """
    一个使用YOLO和SAHI进行切片推理、裁剪和可视化的工具类。
    【多目标类别处理版】
    """
    def __init__(self, model_path: str, output_path: str = './output', device: str = '0'):
        print(f"正在使用SAHI加载YOLO模型: {model_path}")
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print(f"✅ 检测到CUDA，将使用设备: {self.device}")
            else:
                self.device = torch.device("cpu")
                print("⚠️ 未检测到CUDA，将使用CPU。")

            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=model_path,
                confidence_threshold=0.3,
                device=self.device,
            )
            self.class_names = self.detection_model.model.names
            self.class_name_to_id = {v: k for k, v in self.class_names.items()}
            print(f"✅ 模型加载成功。可检测类别: {list(self.class_names.values())}")

            # --- 为绘制功能初始化颜色 ---
            np.random.seed(42)
            self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
            
            self.output_path = output_path
            self.file_counter = self._get_dir_file_number(self.output_path)

        except Exception as e:
            print(f"❌ 错误: 无法加载模型。错误详情: {e}")
            self.detection_model = None

    def _get_dir_file_number(self, directory: str) -> int:
        os.makedirs(directory, exist_ok=True)
        max_num = 0
        print(f"正在扫描目录 '{directory}' 以确定起始编号...")
        try:
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    try:
                        num_str = os.path.splitext(filename)[0]
                        if num_str.startswith("crop_") or num_str.startswith("draw_"): num_str = num_str[5:]
                        num = int(num_str)
                        if num > max_num: max_num = num
                    except ValueError: continue
        except Exception as e: print(f"扫描目录时发生错误: {e}")
        print(f"扫描完成。最大文件编号为: {max_num}。")
        return max_num

    def infer_with_sahi(self, 
                        frame: np.ndarray, 
                        slice_size: int = 640, 
                        overlap_ratio: float = 0.2, 
                        conf_threshold: float = 0.5) -> list[dict]:
        if not self.detection_model: return []
        sahi_result = get_sliced_prediction(
            frame, self.detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio, 
            overlap_width_ratio=overlap_ratio,
            postprocess_match_threshold=conf_threshold,
            # postprocess_class_threshold=conf_threshold
            verbose=2
        )
        
        predictions_list = []
        for pred in sahi_result.object_prediction_list:
            predictions_list.append({
                "box": [int(coord) for coord in pred.bbox.to_xyxy()],
                "score": pred.score.value,
                "class_id": pred.category.id,
                "class_name": pred.category.name
            })
        print(f"SAHI 推理完成。检测到 {len(predictions_list)} 个目标。")
        print(f"检测到的目标: {predictions_list}")
        return predictions_list
    
    def crop_targets(self, predictions: list[dict], image: np.ndarray, target_class_ids: list[int]) -> None:
        """
        从预测结果列表中裁剪出指定类别列表中的目标并保存。

        Args:
            predictions (list[dict]): 来自 infer_with_sahi 方法的预测结果列表。
            image (np.ndarray): 原始的、未被修改的图像帧。
            target_class_ids (list[int]): 要裁剪的目标的类别ID列表。如果为[-1], 则裁剪所有检测到的目标。
        """
        should_crop_all = -1 in target_class_ids
        
        for pred in predictions:
            # 检查当前预测的类别是否在目标列表中，或者是否需要裁剪所有
            if should_crop_all or pred['class_id'] in target_class_ids:
                x1, y1, x2, y2 = pred['box']
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(image.shape[1], x2), min(image.shape[0], y2)
                
                if x1_c < x2_c and y1_c < y2_c:
                    cropped_object = image[y1_c:y2_c, x1_c:x2_c]
                    self.file_counter += 1
                    filename = f"crop_{self.file_counter}.jpg"
                    save_path = os.path.join(self.output_path, filename)
                    cv2.imwrite(save_path, cropped_object)
                    print(f"✂️ 已裁剪图片数量: {self.file_counter}", end='\r')

    def draw_targets(self, predictions: list[dict], image: np.ndarray, target_class_ids: list) -> np.ndarray:
        """
        根据预测结果在图像上绘制指定类别列表中的边界框。

        Args:
            predictions (list[dict]): 来自 infer_with_sahi 方法的预测结果列表。
            image (np.ndarray): 原始的OpenCV图像帧（BGR格式）。
            target_class_ids (list): 要绘制的目标类别ID列表。如果为[-1], 则绘制所有检测到的目标。

        Returns:
            np.ndarray: 经过绘制处理后的新图像帧。
        """
        display_image = image.copy()
        should_draw_all = -1 in target_class_ids
        print(f"target_class_ids: {target_class_ids}___{should_draw_all}")

        for pred in predictions:
            # 检查当前预测的类别是否在目标列表中，或者是否需要绘制所有
            if should_draw_all or pred['class_id'] in target_class_ids:
                box = pred["box"]
                score = pred["score"]
                class_id = pred["class_id"]
                class_name = pred["class_name"]
                
                x1, y1, x2, y2 = box
                color = self.colors[class_id].tolist()
                label = f'ID_{class_id}:{class_name} {score:.2f}'
                print(f"绘制: {label}")
                
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_image, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
                cv2.putText(display_image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # return display_image
                # --- 保存 ---
        try:
            # output_dir = os.path.dirname(self.output_path)
            # if output_dir:
            print(f"output_path: {self.output_path}")
            os.makedirs(self.output_path, exist_ok=True)
            self.file_counter += 1
            filename = f"draw_{self.file_counter}.jpg"
            save_path = os.path.join(self.output_path, filename)
            
            print(f"正在保存图片到: {save_path}")
            cv2.imwrite(save_path, display_image)
            print(f"✅ 结果已成功保存到: {self.output_path}") # 可以选择性地保留此打印语句
            return save_path
        except Exception as e:
            print(f"❌ 保存图片时发生错误到 '{self.output_path}': {e}")
            return ""
    



import os
from queue import Queue
from utils.input_handler import InputStreamHandler # 假设这个文件存在
# from sahi_processor import SahiProcessor
import cv2

if __name__ == '__main__':
    # 1. --- 参数配置 ---
    MODEL_PATH = r'./jg_project/v001-epoch100-aug/weights/best.pt' # 您的检测模型
    # INPUT_SOURCE = r'./datasets/test-video/mp4/your_video.mp4' # 您的视频源
    INPUT_SOURCE = r'./datasets/test-video/test0/mp4/343205-202507201624-202507201629.mp4'
    
    OUTPUT_PATH = r'./output_sahi_draw'
    # OUTPUT_PATH = r'./output_sahi_crop'
    DEVICE = '3'

    # --- SAHI 和 推理参数 ---
    SLICE_SIZE = 640
    OVERLAP_RATIO = 0.2
    CONF_THRESHOLD = 0.45

    # --- 功能控制参数 ---
    # 【新】要裁剪的目标类别ID列表。[-1]代表所有。
    TARGET_IDS_TO_CROP = [1] # 例如，只裁剪ID为1的 'bus'
    # TARGET_IDS_TO_CROP = [0, 1] # 例如，裁剪ID为0的 'car' 和 ID为1的 'bus'
    # TARGET_IDS_TO_CROP = [-1] # 裁剪所有检测到的物体

    # 【新】要绘制的目标类别ID列表。[-1]代表所有。
    # TARGET_IDS_TO_DRAW = [-1] # 在视频上绘制所有检测到的物体
    TARGET_IDS_TO_DRAW = [1] # 只在视频上绘制ID为1的 'bus'
    
    DISPLAY_WINDOW = False # 是否实时显示带框的视频

    # 2. --- 初始化处理器和视频流 ---
    print("===== 开始初始化 =====")
    sahi_proc = SahiProcessor(MODEL_PATH, OUTPUT_PATH, DEVICE)
    
    frame_queue = Queue()
    camera = InputStreamHandler(INPUT_SOURCE, frame_queue)
    camera.run()
    
    print('---------- 开始处理循环 ----------')
    try:
        while True:
            image = frame_queue.get()
            # if image is None: break

            # 1. 使用SAHI进行切片推理
            predictions = sahi_proc.infer_with_sahi(
                frame=image, 
                slice_size=SLICE_SIZE,
                overlap_ratio=OVERLAP_RATIO,
                conf_threshold=CONF_THRESHOLD
            )
            
            # 2. 如果有预测结果，则执行裁剪
            # if predictions and TARGET_IDS_TO_CROP:
            #     sahi_proc.crop_targets(
            #         predictions=predictions, 
            #         image=image, 
            #         target_class_ids=TARGET_IDS_TO_CROP
            #     )
            if predictions and TARGET_IDS_TO_DRAW:
                sahi_proc.draw_targets(
                    predictions=predictions, 
                    image=image, 
                    target_class_ids=TARGET_IDS_TO_DRAW
                )
            
            # # 3. 如果需要显示窗口，则执行绘制
            # if DISPLAY_WINDOW:
            #     annotated_frame = sahi_proc.draw_targets(
            #         predictions=predictions,
            #         image=image,
            #         target_class_ids=TARGET_IDS_TO_DRAW
            #     )
            #     cv2.imshow("SAHI Multi-Target Inference", annotated_frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         print("\n用户手动终止。")
            #         break

    except KeyboardInterrupt:
        print("\n =CTRL= 检测到键盘中断! 正在退出...")
    except Exception as e:
        print(f"\n =ERROR= 发生异常! 原因: {e}")
    finally:
        camera.stop()
        # if DISPLAY_WINDOW:
        #     cv2.destroyAllWindows()

    print('\n =^_^= 全部完成, 再见~')