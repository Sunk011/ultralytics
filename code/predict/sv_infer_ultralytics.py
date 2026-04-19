import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("/home/sk/project/ultralytics/code/predict/car_vis_v3.pt")
# model = YOLO("/home/sk/project/ultralytics/usage/visdrone_args/11m-1280-b32-e300/weights/best.pt")
image = cv2.imread('5.jpg')
results = model(image, imgsz= (1080,1980), conf=0.2)[0]
detections = sv.Detections.from_ultralytics(results)

# Write detections to txt file
with open('5-04.txt', 'w') as f:
    for i in range(len(detections)):
        # print(detections[i])
        class_name = detections.class_id[i]
        confidence = detections.confidence[i]
        bbox = detections.xyxy[i]
        f.write(f"{class_name} {confidence:.2f} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")



box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()


# # Read detections from txt file
# detections_list = []
# with open('detections.txt', 'r') as f:
#     for line in f:
#         parts = line.strip().split()
#         class_name = parts[0]
#         confidence = float(parts[1])
#         bbox = [float(x) for x in parts[2:]]
#         detections_list.append((class_name, confidence, bbox))

# # Extract data
# class_names = [d[0] for d in detections_list]
# confidences = [d[1] for d in detections_list]
# xyxy = [d[2] for d in detections_list]

# # Create Detections object
# detections = sv.Detections(xyxy=xyxy, confidence=confidences, data={'class_name': class_names})
# labels = [
#     f"{class_name} {confidence:.2f}"
#     for class_name, confidence
#     in zip(detections['class_name'], detections.confidence)
# ]

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections, labels=labels)

cv2.imwrite("5-04.jpg", annotated_image)