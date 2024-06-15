import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Path to the video you want to use (Option 1: Raw String)
video_path = r"video\uk.mp4"

# Open video stream
video = cv2.VideoCapture(video_path)

# Create video output file
output_path = video_path.replace('.mp4', '_detected.mp4')
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame from BGR (OpenCV format) to RGB (PyTorch format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    # Process the results
    detections = results[0].boxes

    for row in detections:
        if row.conf >= 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, row.xyxy[0].tolist())
            label = model.names[int(row.cls)]
            bgr = (0, 255, 0)  # Bounding box color
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            # Convert the tensor value of row.conf to a Python scalar value with .item()
            cv2.putText(frame, f'{label} {row.conf.item():.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

    # Write the frame with annotations to the output file
    out.write(frame)

    # Display the frame with detected objects
    cv2.imshow('YOLOv8 Detection', frame)

    # Exit the program on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
video.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved at {output_path}")
