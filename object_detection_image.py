import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Path to the image you want to use
image_path = "img/kinder-spielen-mit-golden-retriever-gorodenkof.jpg"

# Load the image
img = cv2.imread(image_path)

# Convert the image from BGR (OpenCV format) to RGB (PyTorch format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection
results = model(img_rgb)

# Process the results
detections = results[0].boxes
height, width = img.shape[:2]

detected_objects = []  # List to store detected objects

for row in detections:
    if row.conf >= 0.5:  # Confidence threshold
        x1, y1, x2, y2 = map(int, row.xyxy[0].tolist())
        label = model.names[int(row.cls)]
        detected_objects.append(label)
        bgr = (0, 255, 0)  # Bounding box color (green)
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
        # Convert tensor value of row.conf to Python scalar value with .item()
        cv2.putText(img, f'{label} {row.conf.item():.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

# Save the image with detected objects
output_path = image_path.replace('.jpg', '_detected.jpg')
cv2.imwrite(output_path, img)

# Print the detected objects
print(f'Detected objects: {", ".join(detected_objects)}')

# Display the saved image with detected objects
cv2.imshow('YOLOv8 Detection', img)

# Wait for a key press and close the window on any key press
cv2.waitKey(0)
cv2.destroyAllWindows()
