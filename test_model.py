# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO

# model = YOLO('/home/sk/project/jg_project/car_vis/epoch120_v1/weights/best.pt')
model = YOLO("/home/sk/project/jg_project/cls/test_yolo11m/weights/best.pt")
print(model.names)
