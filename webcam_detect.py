from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")

# Open webcam (0 = default cam)
cap = cv2.VideoCapture(0)

# Set width & height (optional)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model.predict(source=frame, conf=0.25, save=False)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 - Webcam Detection", annotated_frame)

    # Exit on pressing Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
