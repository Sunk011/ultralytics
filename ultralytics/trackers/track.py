# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from functools import partial
from pathlib import Path

import torch

from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Initialize trackers for a predictor object
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if predictor.args.task == "classify":
        raise ValueError("❌ Classification doesn't support 'mode=track'")

    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    predictor._feats = None  # reset in case used earlier
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()
    if cfg.tracker_type == "botsort" and cfg.with_reid and cfg.model == "auto":
        from ultralytics.nn.modules.head import Detect

        if not (
            isinstance(predictor.model.model, torch.nn.Module)
            and isinstance(predictor.model.model.model[-1], Detect)
            and not predictor.model.model.model[-1].end2end
        ):
            cfg.model = "yolo11n-cls.pt"
        else:
            # Register hook to extract input of Detect layer
            def pre_hook(module, input):
                predictor._feats = list(input[0])  # unroll to new list to avoid mutation in forward

            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    # print(f"len predictot.results:{len(predictor.results)}")
    # print("-----------------------------------------------------------")
    # print(f"predictor:\n{predictor}")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(result.path).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (result.obb if is_obb else result.boxes).cpu().numpy()
        # print(f"det: {det}")


        tracks, lost_tmp = tracker.update(det, result.orig_img, getattr(result, "feats", None))
        # print(f"tracks: {tracks}")
        # print(f"lost_tmp: {lost_tmp}")
        # if len(tracks) == 0:
        #     continue
        # idx = tracks[:, -1].astype(int)
        # print(f"idx: {idx}")
        # # print(f"predictor.results[{i}]: {predictor.results[i]}")
        # # print(f"result[idx]: {result[idx]}")
        # predictor.results[i] = result[idx]

        # update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        # print("=" * 100)
        # print(f"update_args: {update_args}")
        # print("=" * 100)
        # predictor.results[i].update(**update_args)
        # print("-" * 100)
        # print(f"update_args: {update_args}")
        # print("-" * 100)

        print(f"shape:\t tracks.shape: {tracks.shape if len(tracks) > 0 else tracks}, lost_tmp.shape: {lost_tmp.shape if len(lost_tmp) > 0 else lost_tmp}")
        # 处理正常跟踪结果
        if len(tracks) > 0:
            idx = tracks[:, -1].astype(int)
            predictor.results[i] = result[idx]
        else:
            # 如果没有正常跟踪结果，跳过或创建空结果
            continue
        
        # 拼接tracks和lost_tmp数据（除了最后一列索引）
        combined_data = []
        
        # 添加正常跟踪数据
        if len(tracks) > 0:
            combined_data.append(tracks[:, :-1])  # 排除最后一列索引
            print(f"tracks[:, :-1]: {tracks[:, :-1]}")
        
        # 添加丢失预测数据
        if len(lost_tmp) > 0:
            combined_data.append(lost_tmp[:, :-1])  # 排除最后一列索引
            print(f"lost_tmp[:, :-1]: {lost_tmp[:, :-1]}")
        
        # 执行拼接
        if len(combined_data) > 0:
            import numpy as np
            all_tracks = np.vstack(combined_data)
            print(f"Combined tracks shape: {all_tracks.shape}")
            print(f"Normal tracks: {len(tracks)}, Lost predictions: {len(lost_tmp) if len(lost_tmp) > 0 else 0}")
        else:
            all_tracks = tracks[:, :-1]  # 如果没有lost_tmp，只使用tracks
        tmp_tracks = tracks[:, :-1]
        print(f"all_tracks shape: {all_tracks.shape if len(all_tracks) > 0 else all_tracks}")
        # 更新检测结果
        print(f"predictor.results[{i}].boxes: {predictor.results[i].boxes.data if hasattr(predictor.results[i], 'boxes') and predictor.results[i].boxes is not None else 'None'}")
        print(f"predictor.results[{i}].id: {predictor.results[i].boxes.id if hasattr(predictor.results[i], 'boxes') and predictor.results[i].boxes is not None else 'None'}")
        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(all_tracks)}
        predictor.results[i].update(**update_args)
        print(f"updated predictor.results[{i}].boxes: {predictor.results[i].boxes.data if hasattr(predictor.results[i], 'boxes') and predictor.results[i].boxes is not None else 'None'}")
        print(f"updated predictor.results[{i}].id: {predictor.results[i].boxes.id if hasattr(predictor.results[i], 'boxes') and predictor.results[i].boxes is not None else 'None'}")




        # tracks, lost_tmp = tracker.update(det, result.orig_img, getattr(result, "feats", None))
        # print(f"tracks: {tracks}")
        # print(f"lost_tmp: {lost_tmp}")
        
        # # 处理正常跟踪结果
        # if len(tracks) > 0:
        #     idx = tracks[:, -1].astype(int)
        #     print(f"tracks idx: {idx}")
        #     predictor.results[i] = result[idx]
        #     print("================+++++++=================================")
        #     print(f"tracks: {tracks}")
        #     print("================+++++++=================================")
            
        #     update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        #     predictor.results[i].update(**update_args)
        
        # # 处理丢失的预测结果
        # if len(lost_tmp) > 0:
        #     from ultralytics.engine.results import Boxes, OBB
        #     import numpy as np
            
        #     print(f"Adding {len(lost_tmp)} lost predictions to results")
            
        #     # 提取预测框信息 [x1, y1, x2, y2, track_id, conf, cls, ...]
        #     pred_coords = lost_tmp[:, :4]     # 坐标
        #     pred_track_id = lost_tmp[:, 4]    # track_id
        #     pred_conf = lost_tmp[:, 5]        # 置信度
        #     pred_cls = lost_tmp[:, 6]         # 类别
            
        #     print(f"lost_tmp coords: {pred_coords}")
        #     print(f"lost_tmp track_ids: {pred_track_id}")
            
        #     if is_obb:
        #         # 对于OBB任务
        #         if lost_tmp.shape[1] >= 8:  # 包含角度信息
        #             pred_angle = lost_tmp[:, 7]
        #             pred_data = np.column_stack([pred_coords, pred_angle, pred_conf, pred_cls])
        #         else:
        #             # 如果没有角度信息，设置为0
        #             pred_data = np.column_stack([pred_coords, np.zeros(len(pred_coords)), pred_conf, pred_cls])
                
        #         # 创建OBB对象
        #         pred_obb_tensor = torch.tensor(pred_data, dtype=torch.float32)
        #         pred_obb = OBB(pred_obb_tensor, result.orig_shape)
                
        #         # 合并现有OBB和预测OBB
        #         if hasattr(predictor.results[i], 'obb') and predictor.results[i].obb is not None:
        #             combined_data = torch.cat([predictor.results[i].obb.data, pred_obb.data])
        #             predictor.results[i].obb = OBB(combined_data, result.orig_shape)
        #         else:
        #             predictor.results[i].obb = pred_obb
                    
        #     else:
        #         # 对于普通检测任务
        #         # 检查现有boxes的结构来匹配列数
        #         if hasattr(predictor.results[i], 'boxes') and predictor.results[i].boxes is not None:
        #             existing_data = predictor.results[i].boxes.data
        #             existing_cols = existing_data.shape[1]
        #             print(f"Existing boxes data shape: {existing_data.shape}")
        #             print(f"Existing boxes columns: {existing_cols}")
                    
        #             # 构建与现有数据相同列数的预测数据
        #             if existing_cols == 7:  # [x1, y1, x2, y2, conf, cls, track_id]
        #                 pred_data = np.column_stack([pred_coords, pred_conf, pred_cls, pred_track_id])
        #             elif existing_cols == 6:  # [x1, y1, x2, y2, conf, cls]
        #                 pred_data = np.column_stack([pred_coords, pred_conf, pred_cls])
        #             else:
        #                 print(f"Warning: Unexpected existing boxes columns: {existing_cols}")
        #                 pred_data = np.column_stack([pred_coords, pred_conf, pred_cls])
        #         else:
        #             # 如果没有现有boxes，使用标准格式 [x1, y1, x2, y2, conf, cls]
        #             pred_data = np.column_stack([pred_coords, pred_conf, pred_cls])
                
        #         pred_boxes_tensor = torch.tensor(pred_data, dtype=torch.float32)
                
        #         print(f"pred_data shape: {pred_data.shape}")
        #         print(f"pred_boxes_tensor: {pred_boxes_tensor}")
                
        #         # 创建Boxes对象
        #         pred_boxes_obj = Boxes(pred_boxes_tensor, result.orig_shape)
                
        #         # 合并现有boxes和预测boxes
        #         if hasattr(predictor.results[i], 'boxes') and predictor.results[i].boxes is not None:
        #             combined_data = torch.cat([predictor.results[i].boxes.data, pred_boxes_tensor])
        #             predictor.results[i].boxes = Boxes(combined_data, result.orig_shape)
        #             print(f"Combined boxes shape: {combined_data.shape}")
        #         else:
        #             predictor.results[i].boxes = pred_boxes_obj
        #             print(f"Created new boxes with shape: {pred_boxes_tensor.shape}")
            
        #     # 添加预测信息标识
        #     if not hasattr(predictor.results[i], 'prediction_info'):
        #         predictor.results[i].prediction_info = {}
            
        #     predictor.results[i].prediction_info.update({
        #         'lost_prediction_count': len(lost_tmp),
        #         'lost_track_ids': pred_track_id.tolist(),
        #         'lost_coordinates': pred_coords.tolist(),
        #         'lost_confidences': pred_conf.tolist(),
        #         'lost_classes': pred_cls.tolist()
        #     })
        
        # print("-" * 100)
        # print(f"Final result boxes count: {len(predictor.results[i].boxes) if hasattr(predictor.results[i], 'boxes') and predictor.results[i].boxes is not None else 0}")
        # print("-" * 100)


def register_tracker(model: object, persist: bool) -> None:
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
