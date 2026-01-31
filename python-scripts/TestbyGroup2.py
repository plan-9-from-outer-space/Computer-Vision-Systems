import cv2
from ultralytics import YOLO
import supervision as sv
import os

for folder_num in range(1, 51):
    folder_path = f"C:/Users/WIFA/Desktop/RECORDS/Video to Image/{folder_num}"
    model_path = os.path.join(folder_path, "best.pt")
    video_path = os.path.join(folder_path, "Test.mp4")
    output_path = os.path.join(folder_path, "output_detected2.mp4")

    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print(f"⚠️ Skipping folder {folder_num}: missing model or video")
        continue

    print(f"▶️ Processing folder {folder_num}...")

    model = YOLO(model_path).to("cuda")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        # Status message
        if len(results[0].boxes) > 0:
            status_message = "SOMETHING FIND"
            status_color = (0, 255, 0)  # Green
        else:
            status_message = "DO NOT FIND"
            status_color = (0, 0, 255)  # Red

        # Display the status message
        cv2.putText(frame, status_message, (width - 380, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display position next to the bounding box
            cv2.putText(frame, f"X: {cx}, Y: {cy}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    print(f"✅ Folder {folder_num}: output saved as output_detected.mp4\n")