import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("C:/Users/WIFA/Desktop/RECORDS/Final/Main/best.pt")  # path to trained model

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0]).with_nms()

    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imshow("Live Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
