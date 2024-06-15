import cv2
from ultralytics import YOLO

# Laden des vortrainierten YOLOv8-Modells
model = YOLO("yolov8n.pt")

# Zugriff auf die Webcam
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if not ret:
        break

    # Konvertiere das Bild von BGR (OpenCV-Format) nach RGB (PyTorch-Format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Führe die Objekterkennung durch
    results = model(img_rgb)

    # Verarbeite die Ergebnisse
    detections = results[0].boxes
    height, width = img.shape[:2]

    detected_objects = []  # Liste zum Speichern der erkannten Objekte

    for row in detections:
        if row.conf.item() >= 0.5:  # Konfidenzschwelle
            x1, y1, x2, y2 = map(int, row.xyxy[0].tolist())
            label = model.names[int(row.cls.item())]
            detected_objects.append(label)
            bgr = (0, 255, 0)  # Bounding Box Farbe
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(img, f'{label} {row.conf.item():.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

    # Zeige das Bild mit den erkannten Objekten
    cv2.imshow('YOLOv8 Detection', img)

    # Gebe die erkannten Objekte namentlich aus
    print(f'Detected objects: {", ".join(detected_objects)}')

    # Beende das Programm bei Tastendruck 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
