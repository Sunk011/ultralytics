from ultralytics import YOLO

model = YOLO("/home/sk/project/tensorRT/ultralytics/weights/yolo11m-obj365-640-Pretrain.pt")

# TensorRT FP32
out = model.export(format="engine", imgsz=1024, dynamic=True, verbose=False, batch=8, workspace=2)

# # TensorRT FP16
# out = model.export(format="engine", imgsz=1024, dynamic=True, verbose=False, batch=8, workspace=2, half=True)

# # TensorRT INT8 with calibration `data` (i.e. COCO, ImageNet, or DOTAv1 for appropriate model task)
# out = model.export(
#     format="engine", imgsz=1024, dynamic=True, verbose=False, batch=8, workspace=2, int8=True, data="coco8.yaml"
# )