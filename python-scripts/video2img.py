import cv2
import os

def extract_frames(video_path, output_folder, interval_sec=0.5):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Duration: {duration:.2f} seconds, FPS: {fps}, Total Frames: {total_frames}")

    count = 0
    current_sec = 0

    while current_sec <= duration:
        # Move to the desired timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        count += 1
        current_sec += interval_sec

    cap.release()
    print("Done.")

# Example usage
video_path = "C:/Users/WIFA/Desktop/RECORDS/Video to Image/50/Test.mp4"            # Path to your input video
output_folder = "C:/Users/WIFA/Desktop/RECORDS/Video to Image/50"     # Folder to save images
extract_frames(video_path, output_folder, interval_sec=0.5)
